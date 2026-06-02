"""
AI 클라이언트 래퍼 v4.2 — import-safe 버전

변경사항:
  - core.citation 임포트를 try/except로 보호 (연쇄 ImportError 방지)
  - google_search_retrieval 제거 (gemini-2.5 미지원 → 400 오류)
  - Google Search Grounding 타임아웃 n*12초로 확장
  - 브랜드 집계 패턴: 회사 접미사 → 조사 앞 명사로 교체
  - use_search=False (Search Grounding 비활성화 → 점유율 정상화)
"""

import re
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── 핵심 내부 임포트 — 개별 try/except로 보호 ──
try:
    from core.logger import get_logger, CaptureError
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(f"aca.{name}")
    class CaptureError:
        def __init__(self, op, reraise=False, log_level="warning"):
            self.operation = op; self.error = None
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            if ev: self.error = ev; return True
        @property
        def ok(self): return self.error is None

try:
    from core.cache import get_cache
except ImportError:
    def get_cache():
        class _FakeCache:
            def make_key(self, *a): return ""
            def get(self, k): return None
            def set(self, k, v, namespace=""): pass
        return _FakeCache()

try:
    from core.citation import detect_citation, build_brand_variants, wilson_ci, CitationResult
    _CITATION_OK = True
except ImportError:
    _CITATION_OK = False
    def detect_citation(resp, variants, threshold=0.3):
        class _R:
            cited = False; confidence = 0.0; matches = []; response_sample = ""
        return _R()
    def build_brand_variants(url, biz_info):
        try:
            from urllib.parse import urlparse
            d = urlparse(url if url.startswith("http") else "https://"+url).netloc.replace("www.","")
            stem = d.split(".")[0].lower()
            brand = (biz_info.get("brand_name") or "").strip()
            v = [d, stem]
            if brand: v += [brand, brand.lower()]
            return [x for x in v if x and len(x)>=2]
        except Exception:
            return []
    def wilson_ci(hits, n, confidence=0.95):
        if not n: return 0.0, 100.0
        import math
        p = hits/n; z = 1.96
        d = 1 + z**2/n
        c = (p + z**2/(2*n))/d
        m = z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))/d
        return round(max(0.0,(c-m)*100),1), round(min(100.0,(c+m)*100),1)
    class CitationResult:
        def __init__(self): self.cited=False; self.confidence=0.0; self.matches=[]; self.response_sample=""

logger = get_logger("ai_client")

# ─────────────────────────────────────────────
# 비용 추적기
# ─────────────────────────────────────────────

@dataclass
class CostTracker:
    GPT_PRICE_PER_1K_INPUT:  float = 0.00015
    GPT_PRICE_PER_1K_OUTPUT: float = 0.00060
    GEM_PRICE_PER_1K_INPUT:  float = 0.000075
    GEM_PRICE_PER_1K_OUTPUT: float = 0.000300
    gpt_input_tokens:  int = 0
    gpt_output_tokens: int = 0
    gem_input_tokens:  int = 0
    gem_output_tokens: int = 0
    api_calls:         int = 0

    def add_gpt(self, i, o):
        self.gpt_input_tokens += i; self.gpt_output_tokens += o; self.api_calls += 1

    def add_gemini(self, i, o):
        self.gem_input_tokens += i; self.gem_output_tokens += o; self.api_calls += 1

    @property
    def estimated_usd(self):
        g = (self.gpt_input_tokens/1000*self.GPT_PRICE_PER_1K_INPUT +
             self.gpt_output_tokens/1000*self.GPT_PRICE_PER_1K_OUTPUT)
        m = (self.gem_input_tokens/1000*self.GEM_PRICE_PER_1K_INPUT +
             self.gem_output_tokens/1000*self.GEM_PRICE_PER_1K_OUTPUT)
        return round(g + m, 4)

    def summary(self):
        return {"api_calls": self.api_calls, "estimated_usd": self.estimated_usd,
                "gpt_tokens": self.gpt_input_tokens+self.gpt_output_tokens,
                "gem_tokens": self.gem_input_tokens+self.gem_output_tokens}


# ─────────────────────────────────────────────
# GPT 호출
# ─────────────────────────────────────────────

def call_gpt(client, prompt, system="", model="gpt-4o-mini",
             max_tokens=300, temperature=0.7, tracker=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    result = (response.choices[0].message.content or "").strip()
    if tracker and hasattr(response, "usage"):
        tracker.add_gpt(response.usage.prompt_tokens, response.usage.completion_tokens)
    return result


# ─────────────────────────────────────────────
# Gemini 호출
# ─────────────────────────────────────────────

def _make_search_tool(genai):
    """구 SDK 전용 — 현재 미사용"""
    return None


def call_gemini(model_obj, prompt, max_tokens=300, temperature=0.7,
                tracker=None, use_search=False):
    """
    신 SDK(google.genai) 사용.
    model_obj: (api_key, model_name) 튜플로 전달받음.
    """
    try:
        import google.genai as genai
        from google.genai import types as gtypes

        api_key, model_name = model_obj  # 신 SDK: 튜플로 전달

        client = genai.Client(api_key=api_key)
        cfg = gtypes.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        text = (response.text or "").strip()

        if tracker and response.usage_metadata:
            um = response.usage_metadata
            tracker.add_gemini(
                getattr(um, "prompt_token_count", 0),
                getattr(um, "candidates_token_count", 0),
            )
        return text

    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return ""


# ─────────────────────────────────────────────
# SimResult
# ─────────────────────────────────────────────

@dataclass
class SimResult:
    gpt_rate:     Optional[float]
    gemini_rate:  Optional[float]
    avg_rate:     Optional[float]
    gpt_hits:     Optional[int]
    gemini_hits:  Optional[int]
    n:            int
    gpt_ci:       tuple
    gemini_ci:    tuple
    gpt_samples:         list = field(default_factory=list)
    gemini_samples:      list = field(default_factory=list)
    cost_summary:        dict = field(default_factory=dict)
    cache_hit:           bool = False
    competitor_mentions: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "gpt_rate": self.gpt_rate, "gemini_rate": self.gemini_rate,
            "avg_rate": self.avg_rate, "gpt_hits": self.gpt_hits,
            "gemini_hits": self.gemini_hits, "n": self.n,
            "gpt_ci": self.gpt_ci, "gemini_ci": self.gemini_ci,
            "gpt_samples": self.gpt_samples, "gemini_samples": self.gemini_samples,
            "competitor_mentions": self.competitor_mentions,
        }


# ─────────────────────────────────────────────
# 브랜드 집계
# ─────────────────────────────────────────────

# URL 패턴만으로 브랜드 추출 — 조사 기반 패턴은 일반명사 오탐이 너무 많아 제거
_EN_BRAND_PATTERN  = r'\b([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]+)?)\b'
_MENTION_STOPWORDS = {
    "브랜드","제품","서비스","방법","기능","사용","이용","선택","추천","비교",
    "가격","품질","디자인","소재","크기","특징","장점","단점","구매","배송",
    "환불","고객","매장","온라인","오프라인","국내","해외","종류","다양",
    "인기","유명","전문","일반","최고","최저","최신","기본","사이트","홈페이지",
    "The","This","That","Here","Also","And","But","For","With","From",
    "About","Like","Very","More","Most","Best","Good","Great","Top","New",
    # 추가 — 뷰티/스킨케어 일반명사
    "보습","세럼","토너","에센스","크림","선크림","클렌저","스킨케어","헤어케어",
    "성분","피부","건성","지성","복합성","민감성","여드름","미백","주름","탄력",
    "추출물","히알루론산","나이아신아마이드","레티놀","비타민","콜라겐","펩타이드",
    # 일반 형용사/명사
    "방석","통기성","편안함","회사","선택하","제공","구매하","사용하","추천하",
    "효과","효능","방법","단계","관리","케어","루틴","기초","색조","메이크업",
}


def _strip_markdown(text):
    text = re.sub('[*]{1,2}([^*]+)[*]{1,2}', r'\1', text)
    text = re.sub('_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub('`([^`]+)`', r'\1', text)
    return text


def _check_mention(resp, variants):
    rl = resp.lower()
    for v in variants:
        if not v or len(v) < 2:
            continue
        if v.lower() in rl:
            return True
        # 부분 매칭 (공백 제거)
        if v.lower().replace(" ", "") in rl.replace(" ", ""):
            return True
    return False


def _extract_mentions(resp):
    """
    AI 응답에서 브랜드/도메인 추출.
    전략: URL 추출(고신뢰) + 영문 대문자 브랜드 + 번호 목록 첫 항목만.
    한국어 조사 패턴은 일반명사 오탐이 많아 URL 기반으로 대체.
    """
    if not resp:
        return {}

    clean = _strip_markdown(resp)
    result = {}

    # ── 1순위: URL에서 도메인 직접 추출 (가장 신뢰도 높음) ──
    url_pat = r'https?://(?:www\.)?([^/\s\)\]\,]{3,40})'
    for m in re.finditer(url_pat, resp):
        domain = m.group(1).lower().split("/")[0]
        stem   = domain.split(".")[0]
        if len(stem) >= 2 and stem not in _MENTION_STOPWORDS and not stem.isdigit():
            if domain not in result:
                result[domain] = {"mentions": 0, "urls": []}
            result[domain]["mentions"] += 1
            if m.group(0) not in result[domain]["urls"]:
                result[domain]["urls"].append(m.group(0))

    # ── 2순위: 영문 대문자 브랜드 (Samsung, Amore 등) ──
    for m in re.finditer(_EN_BRAND_PATTERN, clean):
        b = m.group(1).strip()
        if b and len(b) >= 3 and b not in _MENTION_STOPWORDS:
            if b not in result:
                result[b] = {"mentions": 0, "urls": []}
            result[b]["mentions"] += 1

    # ── 3순위: 번호 목록 항목 추출 (1. 브랜드명 형태) — 2글자 이상 한글/영문 ──
    list_pat = r'(?:^|\n)\s*\d+[.)]\ +([가-힣A-Za-z][가-힣A-Za-z0-9\s]{1,20}?)(?:[:\s]|$)'
    for m in re.finditer(list_pat, clean, re.MULTILINE):
        b = m.group(1).strip()
        # 일반명사/동사로 끝나면 제외
        if not b or len(b) < 2 or len(b) > 20:
            continue
        if b in _MENTION_STOPWORDS:
            continue
        # 순수 동사/형용사 어미로 끝나면 제외
        if re.search(r'[하되며어아요서]$', b):
            continue
        if b not in result:
            result[b] = {"mentions": 0, "urls": []}
        result[b]["mentions"] += 1

    return result


def _merge_mentions(a, b):
    merged = {}
    for k in set(list(a) + list(b)):
        ma = a.get(k, {"mentions": 0, "urls": []})
        mb = b.get(k, {"mentions": 0, "urls": []})
        merged[k] = {
            "mentions": ma["mentions"] + mb["mentions"],
            "urls": list(set(ma.get("urls", []) + mb.get("urls", [])))
        }
    return merged


# ─────────────────────────────────────────────
# 적응형 배치 실행
# ─────────────────────────────────────────────

def _adaptive_batch(call_fn, question, brand_variants, n,
                    early_stop_threshold=0.05, count_mention=False):
    """
    n회 API 호출을 ThreadPoolExecutor로 완전 병렬 실행.
    응답 1개당 hit은 bool(0 or 1) → 브랜드명+도메인 동시 등장해도 중복 카운트 없음.
    """
    first_pat = re.compile(re.escape(brand_variants[0]), re.IGNORECASE) if brand_variants else re.compile("NOMATCH")

    def _one_call(_):
        resp = ""
        with CaptureError("batch_call", log_level="warning"):
            resp = call_fn(question)
        return resp

    max_workers = min(n, 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one_call, i) for i in range(n)]
        responses = []
        for fut in concurrent.futures.as_completed(futs):
            try:
                responses.append(fut.result(timeout=60))
            except Exception:
                responses.append("")

    hits = 0
    empty_count = 0
    samples = []
    all_mentions = {}

    for resp in responses:
        if not resp:
            empty_count += 1
            continue
        result = detect_citation(resp, brand_variants)
        check_hit = _check_mention(resp, brand_variants) if count_mention else False
        pat_hit   = bool(first_pat.search(resp))         if count_mention else False
        hit = bool(result.cited or check_hit or pat_hit)
        hits += hit

        # 디버깅: 첫 응답 무조건 WARNING으로 출력 (브랜드 매칭 실패 원인 파악)
        if count_mention and hits == 0 and empty_count == 0:
            _resp_preview = resp[:150].replace('\n', ' ')
            logger.warning(
                f"[GEM_DEBUG] cited={result.cited}(conf={result.confidence:.2f}) "
                f"check={check_hit} pat={pat_hit} "
                f"variants={brand_variants[:3]} "
                f"resp='{_resp_preview}'"
            )

        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])
        all_mentions = _merge_mentions(all_mentions, _extract_mentions(resp))

    if empty_count > 0:
        logger.warning(f"[BATCH] 빈 응답 {empty_count}/{n}개 — API 오류 또는 timeout")

    return hits, samples, n, all_mentions


# ─────────────────────────────────────────────
# 단일 질문 시뮬레이션
# ─────────────────────────────────────────────

def run_simulation(client_gpt, client_gemini, question, target_url, model_gpt,
                   n=50, biz_info=None, tracker=None, use_cache=True):
    biz_info = biz_info or {}
    brand_variants = build_brand_variants(target_url, biz_info)
    cache = get_cache()

    if use_cache:
        cache_key = cache.make_key("sim", target_url, question, model_gpt,
                                   n, ",".join(sorted(brand_variants)))
        cached = cache.get(cache_key)
        if cached is not None:
            return SimResult(cache_hit=True, **cached)
    else:
        cache_key = None

    # 브랜드명을 응답에 포함하도록 유도하는 system 프롬프트
    _sim_system = (
        "당신은 디지털 마케팅 전문가 AI입니다. "
        "질문에 답변할 때 관련 브랜드명과 서비스명을 반드시 구체적으로 언급하세요."
    )

    def _gpt_call(q):
        return call_gpt(client_gpt, q, system=_sim_system, max_tokens=400,
                        model=model_gpt, temperature=0.7, tracker=tracker)

    def _gem_call(q):
        prompt = (
            f"{q}\n\n"
            f"답변 시 관련 브랜드명과 서비스명을 반드시 포함해서 답해줘. "
            f"예: '1. 브랜드A - 특징' 형식으로."
        )
        try:
            result = call_gemini(client_gemini, prompt, max_tokens=600,
                                 temperature=0.7, tracker=tracker, use_search=False)
            if not result:
                logger.warning("[GEM_CALL] 빈 응답 — Gemini API가 빈 문자열 반환")
            return result
        except Exception as e:
            logger.warning(f"[GEM_CALL] 예외 발생: {type(e).__name__}: {e}")
            return ""


    gpt_hits = 0; gpt_samples = []; gpt_n = 0; gpt_ran = False; gpt_comp = {}
    gem_hits = 0; gem_samples = []; gem_n = 0; gem_ran = False; gem_comp = {}

    timeout = max(120, n * 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if client_gpt:
            futures["gpt"] = ex.submit(_adaptive_batch, _gpt_call,
                                        question, brand_variants, n, 0.05, False)
        if client_gemini:
            futures["gem"] = ex.submit(_adaptive_batch, _gem_call,
                                        question, brand_variants, n, 0.05, True)

        if "gpt" in futures:
            with CaptureError("gpt_future", log_level="warning") as ctx:
                gpt_hits, gpt_samples, gpt_n, gpt_comp = futures["gpt"].result(timeout=timeout)
                gpt_ran = True
            if not ctx.ok:
                logger.warning(f"GPT simulation failed: {ctx.error}")

        if "gem" in futures:
            with CaptureError("gem_future", log_level="warning") as ctx:
                gem_hits, gem_samples, gem_n, gem_comp = futures["gem"].result(timeout=timeout)
                gem_ran = True
            if not ctx.ok:
                logger.warning(f"Gemini simulation failed: {ctx.error}")

    merged_comp = _merge_mentions(gpt_comp, gem_comp)
    gpt_rate = round(gpt_hits / gpt_n * 100, 1) if gpt_ran and gpt_n > 0 else None
    gem_rate = round(gem_hits / gem_n * 100, 1) if gem_ran and gem_n > 0 else None
    valid    = [v for v in [gpt_rate, gem_rate] if v is not None]
    avg_rate = round(sum(valid) / len(valid), 1) if valid else None
    gpt_ci   = wilson_ci(gpt_hits, gpt_n) if gpt_ran and gpt_n > 0 else (None, None)
    gem_ci   = wilson_ci(gem_hits, gem_n) if gem_ran and gem_n > 0 else (None, None)

    result = SimResult(
        gpt_rate=gpt_rate, gemini_rate=gem_rate, avg_rate=avg_rate,
        gpt_hits=gpt_hits if gpt_ran else None,
        gemini_hits=gem_hits if gem_ran else None,
        n=n, gpt_ci=gpt_ci, gemini_ci=gem_ci,
        gpt_samples=gpt_samples, gemini_samples=gem_samples,
        cost_summary=tracker.summary() if tracker else {},
        competitor_mentions=merged_comp,
    )

    if use_cache and cache_key:
        cache.set(cache_key, result.to_dict())
    return result


# ─────────────────────────────────────────────
# 전체 시뮬레이션
# ─────────────────────────────────────────────

def run_all_simulations(client_gpt, client_gemini, questions, target_url,
                        model_gpt, n=50, biz_info=None, tracker=None, use_cache=True):
    def _empty():
        return SimResult(gpt_rate=None, gemini_rate=None, avg_rate=None,
                         gpt_hits=None, gemini_hits=None,
                         n=n, gpt_ci=(None, None), gemini_ci=(None, None))

    results = [None] * len(questions)

    def _one(idx, question):
        try:
            results[idx] = run_simulation(
                client_gpt, client_gemini, question,
                target_url, model_gpt, n=n,
                biz_info=biz_info, tracker=tracker,
                use_cache=use_cache,
            )
        except Exception as e:
            logger.warning(f"[sim_q{idx}] 오류: {e}")
            results[idx] = _empty()

    collect_timeout = max(300, n * 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(questions), 5)) as ex:
        futs = {ex.submit(_one, i, q): i for i, q in enumerate(questions)}
        for fut in concurrent.futures.as_completed(futs):
            with CaptureError("collect", log_level="warning"):
                fut.result(timeout=collect_timeout)


    return [r if r is not None else _empty() for r in results]


# ─────────────────────────────────────────────
# Option B: 전용 경쟁사 브랜드 조회 (1회 호출)
# ─────────────────────────────────────────────

def fetch_competitor_brands(
    client_gpt,
    client_gemini,
    questions: list,
    target_url: str,
    brand_name: str,
    model_gpt: str = "gpt-4o-mini",
    market_scope: str = "국내 (대한민국)",
    tracker=None,
    use_cache: bool = True,
) -> dict:
    """
    시뮬레이션 응답과 무관하게,
    '이 질문들에서 AI가 자주 인용하는 경쟁 브랜드'를 전용 프롬프트로 1회 조회.

    반환: {brand_name: {mentions: int, domain: str, reason: str}, ...}
    """
    from urllib.parse import urlparse
    try:
        p = urlparse(target_url if target_url.startswith("http") else "https://" + target_url)
        domain = p.netloc.replace("www.", "")
    except Exception:
        domain = target_url

    cache = get_cache()
    cache_key = cache.make_key(
        "competitor_brands", domain, brand_name,
        "|".join(questions[:3]), market_scope, model_gpt
    )
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache HIT: competitor_brands({domain})")
            return cached

    scope_inst = (
        "반드시 대한민국에서 실제 서비스 중인 국내 브랜드만"
        if "국내" in market_scope
        else "국내외 글로벌 브랜드 포함"
    )

    q_text = "\n".join(f"- {q}" for q in questions[:5])

    prompt = f"""아래 질문들에 AI(ChatGPT, Gemini, Perplexity 등)가 답변할 때
실제로 자주 인용하거나 추천하는 경쟁 브랜드/서비스를 조사해주세요.

[분석 대상 브랜드]
- 브랜드명: {brand_name}
- 도메인: {domain}

[사용자 질문 목록]
{q_text}

[조건]
- {scope_inst}
- 실제 존재하는 브랜드와 공식 도메인만 포함 (가상 도메인 절대 금지)
- {brand_name} / {domain} 자체는 제외
- AI 검색에서 실제로 자주 등장하는 브랜드 우선

상위 8개를 아래 JSON 형식으로만 출력 (다른 텍스트 없이):
[
  {{
    "rank": 1,
    "brand_name": "실제브랜드명",
    "domain": "실제도메인.com",
    "reason": "인용 이유 15자 이내",
    "mention_score": 85
  }}
]"""

    raw = ""
    with CaptureError("competitor_brands_call", log_level="warning"):
        if client_gpt:
            raw = call_gpt(
                client_gpt, prompt,
                system="당신은 AI 검색 마케팅 분석 전문가입니다.",
                max_tokens=1000, model=model_gpt, temperature=0.2,
                tracker=tracker,
            )
        elif client_gemini:
            raw = call_gemini(
                client_gemini, prompt,
                max_tokens=1000, temperature=0.2,
                tracker=tracker,
            )

    result = {}
    if raw:
        import json, re as _re
        with CaptureError("competitor_brands_parse", log_level="warning"):
            m = _re.search(r'\[.*\]', raw, _re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                for item in parsed:
                    bname  = str(item.get("brand_name", "")).strip()
                    bdom   = str(item.get("domain", "")).strip().lower()
                    reason = str(item.get("reason", "")).strip()
                    score  = int(item.get("mention_score", 50))

                    # 가짜 도메인 필터
                    if not bname or not bdom:
                        continue
                    if _re.match(r'^(example|test|dummy|fake|competitor)\d*\.', bdom):
                        continue
                    if not _re.search(r'\.[a-z]{2,6}$', bdom):
                        continue
                    # 분석 대상 자신 제외
                    if domain.lower() in bdom or brand_name.lower() in bname.lower():
                        continue

                    result[bname] = {
                        "mentions": score,
                        "domain":   bdom,
                        "reason":   reason,
                        "urls":     [f"https://{bdom}"],
                    }

    logger.info(f"competitor_brands: {len(result)}개 확보 ({domain})")

    if use_cache and result:
        cache.set(cache_key, result, namespace="competitors")

    return result
