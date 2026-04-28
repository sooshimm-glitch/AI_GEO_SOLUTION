"""
AI 클라이언트 래퍼 v4.3

수정 사항:
  - run_simulation: ThreadPoolExecutor를 try/finally + shutdown(wait=False) 로 교체
    → timeout 후 executor가 block 없이 즉시 반환
  - run_all_simulations: n 상한선 100 적용, wait() 패턴으로 교체
    → pending future를 cancel 처리해 메모리 누수·hang 방지
  - import-safe 버전 유지
  - google_search_retrieval 제거 (gemini-2.5 미지원 → 400 오류)
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
    try:
        return [genai.protos.Tool(google_search=genai.protos.GoogleSearch())]
    except Exception:
        return None


def call_gemini(model_obj, prompt, max_tokens=300, temperature=0.7,
                tracker=None, use_search=False):
    import google.generativeai as genai

    search_tools = _make_search_tool(genai) if use_search else None

    if search_tools:
        gen_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    else:
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature)

    response = None
    try:
        if search_tools:
            response = model_obj.generate_content(
                prompt, tools=search_tools, generation_config=gen_config)
        else:
            response = model_obj.generate_content(
                prompt, generation_config=gen_config)

        try:
            text = (response.text or "").strip()
        except (ValueError, AttributeError):
            text = ""
            try:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                text = text.strip()
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        text = ""

    if response and tracker and hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        tracker.add_gemini(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0))
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

_KO_BRAND_PATTERNS = [
    r'(?:^|\n)\s*(?:\d+[.)]\s*|[-•*]\s*)([가-힣a-zA-Z][가-힣a-zA-Z\s]{1,15}?)(?:\s*[-–(:]|\s*\n)',
    r'([가-힣]{2,8})(?=\s*(?:은|는|이|가|을|를|의|에서|에게|으로|로|도|만|부터|까지)\b)',
    r'\(([A-Z][A-Za-z\s]{2,20})\)',
]
_EN_BRAND_PATTERN  = r'\b([A-Z][a-zA-Z]{2,20}(?:\s[A-Z][a-zA-Z]+)?)\b'
_MENTION_STOPWORDS = {
    "브랜드","제품","서비스","방법","기능","사용","이용","선택","추천","비교",
    "가격","품질","디자인","소재","크기","특징","장점","단점","구매","배송",
    "환불","고객","매장","온라인","오프라인","국내","해외","종류","다양",
    "인기","유명","전문","일반","최고","최저","최신","기본","사이트","홈페이지",
    "The","This","That","Here","Also","And","But","For","With","From",
    "About","Like","Very","More","Most","Best","Good","Great","Top","New",
}
_KO_JOSA_ENDINGS = ('으', '에', '을', '를', '이', '의', '은', '는', '가', '도', '만')


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
        if v.lower().replace(" ", "") in rl.replace(" ", ""):
            return True
    return False


def _extract_mentions(resp):
    if not resp:
        return {}

    clean = _strip_markdown(resp)

    result = {}
    found = []
    for pat in _KO_BRAND_PATTERNS:
        for m in re.findall(pat, clean, re.MULTILINE):
            b = m.strip() if isinstance(m, str) else m
            if b and len(b) >= 2 and b not in _MENTION_STOPWORDS:
                found.append(b)
    for m in re.finditer(_EN_BRAND_PATTERN, clean):
        b = m.group(1).strip()
        if b and len(b) >= 3 and b not in _MENTION_STOPWORDS:
            found.append(b)
    for b in found:
        if b.endswith(_KO_JOSA_ENDINGS) or len(b) < 2 or b.isdigit():
            continue
        if len(b) > 12:
            continue
        result[b] = {"mentions": result.get(b, {}).get("mentions", 0) + 1, "urls": []}

    url_pat = r'https?://[^\s\)\]\,]+'
    for u in re.findall(url_pat, resp):
        for b in result:
            if b.lower().replace(" ", "") in u.lower() and u not in result[b]["urls"]:
                result[b]["urls"].append(u)
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
    samples = []
    probe_n = min(10, n)
    probe_hits = 0
    all_mentions = {}

    for _ in range(probe_n):
        resp = ""
        with CaptureError("probe", log_level="debug"):
            resp = call_fn(question)
        if not resp:
            continue
        result = detect_citation(resp, brand_variants)
        no_match = 'NOMATCH'
        pat = re.escape(brand_variants[0]) if brand_variants else no_match
        hit = (
            result.cited
            or (count_mention and _check_mention(resp, brand_variants))
            or (count_mention and bool(re.search(pat, resp, re.IGNORECASE)))
        )
        if hit:
            probe_hits += 1
        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])
        all_mentions = _merge_mentions(all_mentions, _extract_mentions(resp))

    probe_rate = probe_hits / probe_n if probe_n > 0 else 0
    lo, hi = wilson_ci(probe_hits, probe_n)
    if (hi < early_stop_threshold * 100) or (lo > (1 - early_stop_threshold) * 100):
        logger.info(f"조기 종료: rate={probe_rate:.1%}")
        return probe_hits + round(probe_rate * (n - probe_n)), samples, n, all_mentions

    hits = probe_hits
    for _ in range(n - probe_n):
        resp = ""
        with CaptureError("full", log_level="debug"):
            resp = call_fn(question)
        if not resp:
            continue
        result = detect_citation(resp, brand_variants)
        pat = re.escape(brand_variants[0]) if brand_variants else 'NOMATCH'
        hit = (
            result.cited
            or (count_mention and _check_mention(resp, brand_variants))
            or (count_mention and bool(re.search(pat, resp, re.IGNORECASE)))
        )
        if hit:
            hits += 1
        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])
        all_mentions = _merge_mentions(all_mentions, _extract_mentions(resp))

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

    def _gpt_call(q):
        return call_gpt(client_gpt, q, max_tokens=300,
                        model=model_gpt, temperature=0.7, tracker=tracker)

    def _gem_call(q):
        return call_gemini(client_gemini, q, max_tokens=1024,
                           temperature=0.7, tracker=tracker, use_search=False)

    gpt_hits = 0; gpt_samples = []; gpt_n = 0; gpt_ran = False; gpt_comp = {}
    gem_hits = 0; gem_samples = []; gem_n = 0; gem_ran = False; gem_comp = {}

    timeout = max(120, n * 4)

    # FIX: try/finally + shutdown(wait=False) — timeout 후 executor가 block하지 않음
    _sim_ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        futures = {}
        if client_gpt:
            futures["gpt"] = _sim_ex.submit(_adaptive_batch, _gpt_call,
                                             question, brand_variants, n, 0.05, False)
        if client_gemini:
            futures["gem"] = _sim_ex.submit(_adaptive_batch, _gem_call,
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
    finally:
        _sim_ex.shutdown(wait=False)

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
    # FIX: 상한선 100 — 사이드바 최대값과 동일, 무한 루프 방지
    n = min(n, 100)

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

    # FIX: try/finally + shutdown(wait=False) + wait() 패턴
    # pending future는 cancel 처리 후 빈 결과로 채워 hang 방지
    _all_ex = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(questions), 5))
    try:
        futs = {_all_ex.submit(_one, i, q): i for i, q in enumerate(questions)}
        done, pending = concurrent.futures.wait(futs, timeout=collect_timeout)
        for fut in pending:
            fut.cancel()
            idx = futs[fut]
            if results[idx] is None:
                logger.warning(f"시뮬레이션 타임아웃 — 질문 {idx} 빈 결과로 대체")
                results[idx] = _empty()
    finally:
        _all_ex.shutdown(wait=False)

    return [r if r is not None else _empty() for r in results]
