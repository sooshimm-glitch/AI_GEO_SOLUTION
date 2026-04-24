"""
AI 클라이언트 래퍼 v4.0

핵심 수정:
1. sim_prompt = question 그대로 전달 (래핑 제거)
2. _adaptive_batch 반환값 통일 (항상 4-tuple)
3. GPT도 4-tuple 언팩킹으로 통일
4. 브랜드 변형 정확히 전달
5. 브랜드 집계 — 한/영 패턴 추출
6. Gemini 마크다운 금지 지시 제거
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
    api_calls:         int = 0

    def add_gpt(self, input_t: int, output_t: int):
        self.gpt_input_tokens  += input_t
        self.gpt_output_tokens += output_t
        self.api_calls         += 1

    def add_gemini(self, input_t: int, output_t: int):
        self.gem_input_tokens  += input_t
        self.gem_output_tokens += output_t
        self.api_calls         += 1

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
    prompt:      str,
    system:      str = "",
    model:       str = "gpt-4o-mini",
    max_tokens:  int = 300,
    temperature: float = 0.7,
    tracker:     Optional[CostTracker] = None,
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

    result = response.choices[0].message.content or ""
    result = result.strip()

    if tracker and hasattr(response, "usage"):
        tracker.add_gpt(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    return result


# ─────────────────────────────────────────────
# Gemini 호출
# ─────────────────────────────────────────────

def call_gemini(
    model_obj,
    prompt:      str,
    max_tokens:  int = 300,
    temperature: float = 0.7,
    tracker:     Optional[CostTracker] = None,
) -> str:
    import google.generativeai as genai

    # 마크다운 금지 지시 제거 — 실제 대화창처럼 자연스럽게
    response = model_obj.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    try:
        text = response.text or ""
        text = text.strip()
    except (AttributeError, ValueError):
        logger.warning("Gemini returned empty/blocked response")
        text = ""

    if tracker and hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        tracker.add_gemini(
            getattr(um, "prompt_token_count",     0),
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
    gpt_samples:  list = field(default_factory=list)
    gemini_samples: list = field(default_factory=list)
    cost_summary: dict = field(default_factory=dict)
    cache_hit:    bool = False
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


def _extract_brand_mentions(resp: str) -> dict:
    """
    AI 답변에서 브랜드명 자동 추출 및 URL 집계.
    반환: {브랜드명: {"mentions": N, "urls": [...]}}
    """
    result = {}
    if not resp:
        return result

    # 한국어 브랜드 패턴 — 업체명 접미사 포함
    ko_pattern = r'[가-힣]{2,}(?:미디어|마케팅|광고|에이전시|컴퍼니|코퍼레이션|솔루션|테크|소프트|시스템|네트웍|커뮤니케이션|파트너스|어드바이저|컨설팅|그룹|홀딩스)'
    # 영어 브랜드 패턴 — 대문자 시작
    en_pattern = r'\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+){0,2}\b'
    # URL 추출
    url_pattern = r'https?://[^\s\)\]\,]+'

    ko_brands = re.findall(ko_pattern, resp)
    en_brands = re.findall(en_pattern, resp)
    urls      = re.findall(url_pattern, resp)

    for b in ko_brands + en_brands:
        b = b.strip()
        if len(b) < 2:
            continue
        if b not in result:
            result[b] = {"mentions": 0, "urls": []}
        result[b]["mentions"] += 1

    # URL을 브랜드에 매핑
    for u in urls:
        for b in result:
            if b.lower() in u.lower():
                if u not in result[b]["urls"]:
                    result[b]["urls"].append(u)

    return result


def _merge_brand_mentions(a: dict, b: dict) -> dict:
    """두 집계 딕셔너리 합산"""
    merged = {}
    for k in set(list(a.keys()) + list(b.keys())):
        ma = a.get(k, {"mentions": 0, "urls": []})
        mb = b.get(k, {"mentions": 0, "urls": []})
        merged[k] = {
            "mentions": ma["mentions"] + mb["mentions"],
            "urls": list(set(ma["urls"] + mb["urls"]))
        }
    return merged


# ─────────────────────────────────────────────
# 적응형 배치 실행
# 반환: (hits, samples, n, brand_mentions_dict)  ← 항상 4-tuple
# ─────────────────────────────────────────────

def _adaptive_batch(
    call_fn:              Callable[[str], str],
    question:             str,
    brand_variants:       list,
    n:                    int,
    early_stop_threshold: float = 0.05,
    count_mention:        bool  = False,
) -> tuple:
    samples       = []
    probe_n       = min(10, n)
    probe_hits    = 0
    all_mentions  = {}  # 누적 브랜드 집계

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

        # 브랜드 집계
        mentions = _extract_brand_mentions(resp)
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

        mentions = _extract_brand_mentions(resp)
        all_mentions = _merge_brand_mentions(all_mentions, mentions)

    return hits, samples, n, all_mentions


# ─────────────────────────────────────────────
# 단일 질문 시뮬레이션
# ─────────────────────────────────────────────

def run_simulation(
    client_gpt,
    client_gemini,
    question:   str,
    target_url: str,
    model_gpt:  str,
    n:          int = 50,
    biz_info:   dict = None,
    tracker:    Optional[CostTracker] = None,
    use_cache:  bool = True,
) -> SimResult:
    biz_info       = biz_info or {}
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

    # ── 질문 그대로 전달 (래핑 없이 실제 대화창처럼) ──
    sim_question = question

    def _gpt_call(q: str) -> str:
        return call_gpt(
            client_gpt, q,
            max_tokens=300,
            model=model_gpt,
            temperature=0.7,
            tracker=tracker,
        )

    def _gem_call(q: str) -> str:
        return call_gemini(
            client_gemini, q,
            max_tokens=300,
            temperature=0.7,
            tracker=tracker,
        )

    gpt_hits, gpt_samples, gpt_n = 0, [], 0
    gem_hits, gem_samples, gem_n = 0, [], 0
    gpt_ran = False
    gem_ran = False
    gpt_comp = {}
    gem_comp = {}

    timeout = min(120, max(60, n * 2))

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

    # 경쟁사 언급 합산
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
        gpt_hits=gpt_hits    if gpt_ran else None,
        gemini_hits=gem_hits if gem_ran else None,
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
    questions:  list,
    target_url: str,
    model_gpt:  str,
    n:          int = 50,
    biz_info:   dict = None,
    tracker:    Optional[CostTracker] = None,
    use_cache:  bool = True,
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
    collect_timeout = min(300, max(90, n * 3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_sim_one, i, q): i for i, q in enumerate(questions)}
        for fut in concurrent.futures.as_completed(futs):
            with CaptureError("sim_future_collect", log_level="warning"):
                fut.result(timeout=collect_timeout)

    results = [r if r is not None else _empty_result() for r in results]
    return results
