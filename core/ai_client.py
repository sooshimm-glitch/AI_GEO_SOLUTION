"""
AI 클라이언트 래퍼 v4.1

버그 수정:
  [BUG 1] _extract_brand_mentions — ko_pattern이 회사 접미사만 잡아서
           일반 브랜드명("이케아", "한샘", "무인양품" 등)을 전혀 못 탐지
           → 한국어 조사 앞 명사 패턴으로 교체
  [BUG 2] call_gemini — Google Search Grounding 미사용
           → Gemini 웹 UI와 달리 로컬/틈새 브랜드를 학습데이터에만 의존
           → use_search=True 파라미터 추가, 시뮬레이션은 항상 search=True
  [BUG 3] (citation.py 참조) build_brand_variants 하이픈 도메인 미분리
           → avahair-avenue 에서 avahair, avenue 개별 추출 안 됨
"""

import re
import json
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, Callable
from core.logger import get_logger, CaptureError
from core.cache import get_cache
from core.citation import detect_citation, build_brand_variants, wilson_ci, CitationResult

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
    api_calls: int = 0

    def add_gpt(self, input_t: int, output_t: int):
        self.gpt_input_tokens  += input_t
        self.gpt_output_tokens += output_t
        self.api_calls += 1

    def add_gemini(self, input_t: int, output_t: int):
        self.gem_input_tokens  += input_t
        self.gem_output_tokens += output_t
        self.api_calls += 1

    @property
    def estimated_usd(self) -> float:
        gpt = (self.gpt_input_tokens  / 1000 * self.GPT_PRICE_PER_1K_INPUT +
               self.gpt_output_tokens / 1000 * self.GPT_PRICE_PER_1K_OUTPUT)
        gem = (self.gem_input_tokens  / 1000 * self.GEM_PRICE_PER_1K_INPUT +
               self.gem_output_tokens / 1000 * self.GEM_PRICE_PER_1K_OUTPUT)
        return round(gpt + gem, 4)

    def summary(self) -> dict:
        return {
            "api_calls":     self.api_calls,
            "estimated_usd": self.estimated_usd,
            "gpt_tokens":    self.gpt_input_tokens  + self.gpt_output_tokens,
            "gem_tokens":    self.gem_input_tokens  + self.gem_output_tokens,
        }

# ─────────────────────────────────────────────
# GPT 호출
# ─────────────────────────────────────────────

def call_gpt(
    client,
    prompt: str,
    system: str = "",
    model: str = "gpt-4o-mini",
    max_tokens: int = 300,
    temperature: float = 0.7,
    tracker: Optional[CostTracker] = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = (response.choices[0].message.content or "").strip()

    if tracker and hasattr(response, "usage"):
        tracker.add_gpt(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    return result

# ─────────────────────────────────────────────
# Gemini — Google Search Grounding 지원
# ─────────────────────────────────────────────

def _get_gemini_search_tool(genai):
    """
    Google Search Grounding 도구 반환.
    gemini-2.5 이상: google_search 전용 (google_search_retrieval → 400 오류)
    """
    try:
        return [genai.protos.Tool(google_search=genai.protos.GoogleSearch())]
    except Exception:
        pass
    # google_search_retrieval 은 gemini-2.5에서 지원 안 함 — 사용하지 않음
    return None


def call_gemini(
    model_obj,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    tracker: Optional[CostTracker] = None,
    use_search: bool = False,
) -> str:
    """
    Gemini API 호출.

    use_search=True (기본값: 시뮬레이션에서 항상 True):
      Google Search Grounding 활성화 → Gemini 웹 UI와 동일하게 실시간 검색 참조.
      로컬 비즈니스·최신 정보 인용 정확도 대폭 향상.
      search 활성화 시 temperature는 API 제약으로 전달 불가.
    """
    import google.generativeai as genai

    search_tools = _get_gemini_search_tool(genai) if use_search else None

    if search_tools:
        # search 활성화 시 temperature 미지원
        gen_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    else:
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    response = None
    try:
        if search_tools:
            response = model_obj.generate_content(
                prompt,
                tools=search_tools,
                generation_config=gen_config,
            )
        else:
            response = model_obj.generate_content(
                prompt,
                generation_config=gen_config,
            )
        text = (response.text or "").strip()
    except (ValueError, AttributeError):
        text = ""
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        text = ""

    if response and tracker and hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        tracker.add_gemini(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
        )
    return text

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
    gpt_samples:       list = field(default_factory=list)
    gemini_samples:    list = field(default_factory=list)
    cost_summary:      dict = field(default_factory=dict)
    cache_hit:         bool = False
    competitor_mentions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gpt_rate":            self.gpt_rate,
            "gemini_rate":         self.gemini_rate,
            "avg_rate":            self.avg_rate,
            "gpt_hits":            self.gpt_hits,
            "gemini_hits":         self.gemini_hits,
            "n":                   self.n,
            "gpt_ci":              self.gpt_ci,
            "gemini_ci":           self.gemini_ci,
            "gpt_samples":         self.gpt_samples,
            "gemini_samples":      self.gemini_samples,
            "competitor_mentions": self.competitor_mentions,
        }

# ─────────────────────────────────────────────
# 브랜드 집계 유틸
# ─────────────────────────────────────────────

def _check_brand_mention(resp: str, brand_variants: list) -> bool:
    """브랜드 변형어가 응답에 한 번이라도 등장하면 True"""
    resp_lower = resp.lower()
    for v in brand_variants:
        if v and len(v) >= 2 and v.lower() in resp_lower:
            return True
    return False


# [BUG 1 FIX] 브랜드 추출 패턴 전면 교체
# 기존: ko_pattern = 회사 접미사 한정 → 일반 브랜드명 전혀 못 잡음
# 수정: 한국어 조사/접미사 앞 명사 + 번호 목록 패턴으로 대부분의 브랜드 탐지
_KO_BRAND_PATTERNS = [
    # 번호 목록 패턴: "1. 브랜드명", "1) 브랜드명", "• 브랜드명"
    r'(?:^|\n)\s*(?:\d+[.)]\s*|[-•*]\s*)([가-힣a-zA-Z][가-힣a-zA-Z\s]{1,15}?)(?:\s*[-–(:]|\s*\n)',
    # 조사 앞 한국어 명사: "브랜드는", "브랜드이", "브랜드가", "브랜드을/를"
    r'([가-힣]{2,8})(?=\s*(?:은|는|이|가|을|를|의|에서|에게|으로|로|도|만|부터|까지)\b)',
    # 괄호 안 영문 브랜드명: "(IKEA)", "(MUJI)"
    r'\(([A-Z][A-Za-z\s]{2,20})\)',
]

_EN_BRAND_PATTERN = r'\b([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]+)?)\b'

# 집계에서 제외할 일반 단어
_MENTION_STOPWORDS = {
    # 한국어 일반어 (확장)
    "브랜드", "제품", "서비스", "방법", "기능", "사용", "이용", "선택", "추천", "비교",
    "가격", "품질", "디자인", "소재", "크기", "색상", "특징", "장점", "단점",
    "구매", "배송", "환불", "고객", "매장", "온라인", "오프라인", "국내", "해외",
    "종류", "방석", "소파", "의자", "침대", "쿠션", "인테리어", "가구", "홈",
    "다양", "인기", "유명", "전문", "일반", "최고", "최저", "최신", "기본",
    "사이트", "홈페이지", "플랫폼", "앱", "어플", "쇼핑", "스토어", "마켓",
    # 영어 일반어
    "The", "This", "That", "Here", "There", "Also", "And", "But", "For",
    "With", "From", "About", "Like", "Just", "Very", "More", "Most",
    "Best", "Good", "Great", "High", "Low", "New", "Old", "Big", "Small",
    "Top", "First", "Last", "Main", "Other", "Each", "Some", "Many",
}

# 조사 잘못 포함 후처리 플래그
_KO_JOSA_ENDINGS = ('으', '에', '을', '를', '이', '의', '은', '는', '가', '도', '만')


def _extract_brand_mentions(resp: str) -> dict:
    """
    [BUG 1 FIX] AI 응답에서 브랜드명 자동 추출.

    수정 전: 회사 접미사(마케팅, 솔루션 등) 한정 → 일반 브랜드 전혀 못 잡음
    수정 후: 번호 목록, 조사 앞 명사, 괄호 영문, 대문자 영문 패턴 종합 탐지

    반환: {브랜드명: {"mentions": N, "urls": [...]}}
    """
    result: dict = {}
    if not resp:
        return result

    # URL 추출
    url_pattern = r'https?://[^\s\)\]\,]+'
    urls = re.findall(url_pattern, resp)

    found_brands: list[str] = []

    # 한국어 브랜드 패턴들
    for pat in _KO_BRAND_PATTERNS:
        matches = re.findall(pat, resp, re.MULTILINE)
        for m in matches:
            b = m.strip() if isinstance(m, str) else m[0].strip()
            if b and len(b) >= 2 and b not in _MENTION_STOPWORDS:
                found_brands.append(b)

    # 영어 브랜드 패턴
    for m in re.finditer(_EN_BRAND_PATTERN, resp):
        b = m.group(1).strip()
        if b and len(b) >= 3 and b not in _MENTION_STOPWORDS:
            found_brands.append(b)

    # 집계
    for b in found_brands:
        # 후처리: 조사 일부가 붙은 경우 제거 ("방석으" → 제외)
        if b.endswith(_KO_JOSA_ENDINGS):
            continue
        # 2자 미만 / 순수 숫자 제외
        if len(b) < 2 or b.isdigit():
            continue
        if b not in result:
            result[b] = {"mentions": 0, "urls": []}
        result[b]["mentions"] += 1

    # URL → 브랜드 매핑
    for u in urls:
        for b in result:
            if b.lower().replace(" ", "") in u.lower():
                if u not in result[b]["urls"]:
                    result[b]["urls"].append(u)

    # mentions=0인 항목 제거 (중복 처리 과정에서 발생 가능)
    return {k: v for k, v in result.items() if v["mentions"] > 0}


def _merge_brand_mentions(a: dict, b: dict) -> dict:
    """두 집계 딕셔너리 합산"""
    merged = {}
    for k in set(list(a.keys()) + list(b.keys())):
        ma = a.get(k, {"mentions": 0, "urls": []})
        mb = b.get(k, {"mentions": 0, "urls": []})
        merged[k] = {
            "mentions": ma["mentions"] + mb["mentions"],
            "urls": list(set(ma.get("urls", []) + mb.get("urls", []))),
        }
    return merged

# ─────────────────────────────────────────────
# 적응형 배치 실행 — 항상 4-tuple 반환
# ─────────────────────────────────────────────

def _adaptive_batch(
    call_fn: Callable[[str], str],
    question: str,
    brand_variants: list,
    n: int,
    early_stop_threshold: float = 0.05,
    count_mention: bool = False,
) -> tuple:
    samples      = []
    probe_n      = min(10, n)
    probe_hits   = 0
    all_mentions: dict = {}

    for _ in range(probe_n):
        resp = ""
        with CaptureError("adaptive_probe", log_level="debug"):
            resp = call_fn(question)
        if not resp:
            continue

        result: CitationResult = detect_citation(resp, brand_variants)
        hit = result.cited or (count_mention and _check_brand_mention(resp, brand_variants))
        if hit:
            probe_hits += 1
        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])

        mentions    = _extract_brand_mentions(resp)
        all_mentions = _merge_brand_mentions(all_mentions, mentions)

    probe_rate = probe_hits / probe_n if probe_n > 0 else 0
    lo, hi     = wilson_ci(probe_hits, probe_n)

    # 조기 종료
    if (hi < early_stop_threshold * 100) or (lo > (1 - early_stop_threshold) * 100):
        logger.info(f"조기 종료: rate={probe_rate:.1%}, CI=[{lo:.1f},{hi:.1f}]%")
        remaining_hits = round(probe_rate * (n - probe_n))
        return probe_hits + remaining_hits, samples, n, all_mentions

    hits = probe_hits
    for _ in range(n - probe_n):
        resp = ""
        with CaptureError("adaptive_full", log_level="debug"):
            resp = call_fn(question)
        if not resp:
            continue

        result: CitationResult = detect_citation(resp, brand_variants)
        hit = result.cited or (count_mention and _check_brand_mention(resp, brand_variants))
        if hit:
            hits += 1
        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])

        mentions    = _extract_brand_mentions(resp)
        all_mentions = _merge_brand_mentions(all_mentions, mentions)

    return hits, samples, n, all_mentions

# ─────────────────────────────────────────────
# 단일 질문 시뮬레이션
# ─────────────────────────────────────────────

def run_simulation(
    client_gpt,
    client_gemini,
    question: str,
    target_url: str,
    model_gpt: str,
    n: int = 50,
    biz_info: dict = None,
    tracker: Optional[CostTracker] = None,
    use_cache: bool = True,
) -> SimResult:
    biz_info      = biz_info or {}
    brand_variants = build_brand_variants(target_url, biz_info)
    cache          = get_cache()

    if use_cache:
        cache_key = cache.make_key(
            "sim", target_url, question, model_gpt,
            n, ",".join(sorted(brand_variants))
        )
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache HIT: simulation({question[:30]}...)")
            return SimResult(cache_hit=True, **cached)
    else:
        cache_key = None

    sim_question = question  # 래핑 없이 그대로 전달

    def _gpt_call(q: str) -> str:
        return call_gpt(
            client_gpt, q,
            max_tokens=300,
            model=model_gpt,
            temperature=0.7,
            tracker=tracker,
        )

    # [BUG 2 FIX] Gemini 시뮬레이션 — Google Search Grounding 활성화
    # 이전: use_search 없음 → 학습 데이터만 참조 → 로컬/틈새 브랜드 0% 인용
    # 수정: use_search=True → 실시간 검색으로 Gemini 웹 UI와 동일한 인용 결과
    def _gem_call(q: str) -> str:
        return call_gemini(
            client_gemini, q,
            max_tokens=300,
            temperature=0.7,
            tracker=tracker,
            use_search=True,    # ← Google Search Grounding ON
        )

    gpt_hits, gpt_samples, gpt_n = 0, [], 0
    gem_hits, gem_samples, gem_n = 0, [], 0
    gpt_ran = False
    gem_ran = False
    gpt_comp: dict = {}
    gem_comp: dict = {}

    # Google Search Grounding 활성화 시 호출당 5~15초 소요
    # n=30 기준 최소 300초 필요 → 넉넉하게 설정
    timeout = max(300, n * 12)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if client_gpt:
            futures["gpt"] = ex.submit(
                _adaptive_batch, _gpt_call, sim_question, brand_variants, n, 0.05, False
            )
        if client_gemini:
            futures["gem"] = ex.submit(
                _adaptive_batch, _gem_call, sim_question, brand_variants, n, 0.05, True
            )

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

    merged_comp = _merge_brand_mentions(gpt_comp, gem_comp)

    gpt_rate = round(gpt_hits / gpt_n * 100, 1) if gpt_ran and gpt_n > 0 else None
    gem_rate = round(gem_hits / gem_n * 100, 1) if gem_ran and gem_n > 0 else None
    valid    = [v for v in [gpt_rate, gem_rate] if v is not None]
    avg_rate = round(sum(valid) / len(valid), 1) if valid else None

    gpt_ci = wilson_ci(gpt_hits, gpt_n) if gpt_ran and gpt_n > 0 else (None, None)
    gem_ci = wilson_ci(gem_hits, gem_n) if gem_ran and gem_n > 0 else (None, None)

    result = SimResult(
        gpt_rate=gpt_rate,
        gemini_rate=gem_rate,
        avg_rate=avg_rate,
        gpt_hits=gpt_hits     if gpt_ran else None,
        gemini_hits=gem_hits  if gem_ran else None,
        n=n,
        gpt_ci=gpt_ci,
        gemini_ci=gem_ci,
        gpt_samples=gpt_samples,
        gemini_samples=gem_samples,
        cost_summary=tracker.summary() if tracker else {},
        competitor_mentions=merged_comp,
    )

    if use_cache and cache_key:
        cache.set(cache_key, result.to_dict())

    return result

# ─────────────────────────────────────────────
# 전체 질문 병렬 시뮬레이션
# ─────────────────────────────────────────────

def run_all_simulations(
    client_gpt,
    client_gemini,
    questions: list,
    target_url: str,
    model_gpt: str,
    n: int = 50,
    biz_info: dict = None,
    tracker: Optional[CostTracker] = None,
    use_cache: bool = True,
) -> list:

    def _empty_result() -> SimResult:
        return SimResult(
            gpt_rate=None, gemini_rate=None, avg_rate=None,
            gpt_hits=None, gemini_hits=None,
            n=n, gpt_ci=(None, None), gemini_ci=(None, None),
        )

    results = [None] * len(questions)

    def _sim_one(idx: int, question: str):
        try:
            r = run_simulation(
                client_gpt, client_gemini, question, target_url,
                model_gpt, n=n, biz_info=biz_info,
                tracker=tracker, use_cache=use_cache,
            )
            results[idx] = r
        except Exception as e:
            logger.warning(f"[sim_q{idx}] 오류: {e}")
            results[idx] = _empty_result()

    max_workers     = min(len(questions), 5)
    collect_timeout = max(420, n * 15)  # search 포함 여유 타임아웃

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_sim_one, i, q): i for i, q in enumerate(questions)}
        for fut in concurrent.futures.as_completed(futs):
            with CaptureError("sim_future_collect", log_level="warning"):
                fut.result(timeout=collect_timeout)

    return [r if r is not None else _empty_result() for r in results]
