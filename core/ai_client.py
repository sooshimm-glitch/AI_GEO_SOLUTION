"""
AI 클라이언트 래퍼 — 비용 인식 샘플링 + 적응형 조기 종료

수정 사항 (버그픽스):
1. 캐시 hit 체크 언팩킹 오류 수정 — cache.get() 단일 반환값으로 통일
2. _adaptive_batch 빈 응답 방어 처리
3. 타임아웃 상한선 현실화 (최대 120초)
4. call_gemini safety filter 빈 응답 AttributeError 방어
5. results None 잔존 방어 — 수집 후 None 항목 SimResult로 교체
"""

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
    """세션 내 API 비용 누적 추적 (USD 기준 추정)"""
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

    # Gemini 마크다운 아이콘/이모지 방지 지시를 프롬프트 앞에 주입
    _clean_prompt = (
        "중요: 마크다운 기호(_arrowRight_ 등 _word_, *, **, 이모지, 특수문자) 절대 사용 금지. "
        "순수 텍스트만 출력.\n\n" + prompt
    )

    response = model_obj.generate_content(
        _clean_prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    # BUG FIX: safety filter 등으로 빈 응답 반환 시 AttributeError 방어
    try:
        text = response.text or ""
        text = text.strip()
    except (AttributeError, ValueError):
        # response.text 접근 자체가 예외를 던지는 경우 (blocked response 등)
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
    competitor_mentions: dict = field(default_factory=dict)  # {브랜드명: {mentions:N, urls:[...]}}

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
# 적응형 배치 실행
# ─────────────────────────────────────────────

def _check_brand_mention(resp: str, brand_variants: list) -> bool:
    """
    Gemini 전용 — 브랜드명 단순 언급 체크.
    detect_citation의 threshold(0.3) 통과 여부와 무관하게
    브랜드 변형어가 응답에 한 번이라도 등장하면 True.
    """
    resp_lower = resp.lower()
    for v in brand_variants:
        if v and len(v) >= 2 and v.lower() in resp_lower:
            return True
    return False


def _adaptive_batch(
    call_fn:               Callable[[str], str],
    question:              str,
    brand_variants:        list,
    n:                     int,
    early_stop_threshold:  float = 0.05,
    count_mention:         bool  = False,  # Gemini 전용: 브랜드 언급도 집계
    competitor_names:      list  = None,   # 경쟁사 브랜드명 리스트 (언급 집계용)
) -> tuple:
    samples         = []
    probe_n         = min(10, n)
    probe_hits      = 0
    # 경쟁사 언급 카운터: {브랜드명: {mentions:0, urls:set()}}
    comp_mentions   = {c: {"mentions": 0, "urls": set()} for c in (competitor_names or [])}

    for _ in range(probe_n):
        resp = ""
        with CaptureError("adaptive_probe", log_level="debug"):
            resp = call_fn(question)

        if not resp:
            continue

        result: CitationResult = detect_citation(resp, brand_variants)
        # Gemini: 정식 citation OR 단순 언급 모두 집계
        hit = result.cited or (count_mention and _check_brand_mention(resp, brand_variants))
        if hit:
            probe_hits += 1
        if len(samples) < 3 and (result.response_sample or hit):
            samples.append(result.response_sample or resp[:200])
        # 경쟁사 언급 집계
        if comp_mentions and resp:
            import re as _re
            resp_lower = resp.lower()
            for comp in comp_mentions:
                if comp.lower() in resp_lower:
                    comp_mentions[comp]["mentions"] += 1
                urls = _re.findall(r'https?://[^\s\)\]]+', resp)
                for u in urls:
                    if comp.lower() in u.lower():
                        comp_mentions[comp]["urls"].add(u)

    probe_rate = probe_hits / probe_n if probe_n > 0 else 0
    lo, hi     = wilson_ci(probe_hits, probe_n)

    # 조기 종료 조건
    if (hi < early_stop_threshold * 100) or (lo > (1 - early_stop_threshold) * 100):
        logger.info(f"조기 종료: rate={probe_rate:.1%}, CI=[{lo:.1f},{hi:.1f}]%")
        remaining_hits = round(probe_rate * (n - probe_n))
        comp_result = {k: {"mentions": v["mentions"], "urls": list(v["urls"])} for k,v in comp_mentions.items()}
        return probe_hits + remaining_hits, samples, n, comp_result

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
        # 경쟁사 언급 집계
        if comp_mentions and resp:
            import re as _re
            resp_lower = resp.lower()
            for comp in comp_mentions:
                if comp.lower() in resp_lower:
                    comp_mentions[comp]["mentions"] += 1
                urls = _re.findall(r'https?://[^\s\)\]]+', resp)
                for u in urls:
                    if comp.lower() in u.lower():
                        comp_mentions[comp]["urls"].add(u)

    # comp_mentions urls를 list로 변환
    comp_result = {k: {"mentions": v["mentions"], "urls": list(v["urls"])} for k,v in comp_mentions.items()}
    return hits, samples, n, comp_result


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

    # BUG FIX: cache.get()은 단일 값 반환 — 언팩킹 제거
    if use_cache:
        cache_key = cache.make_key(
            "sim", target_url, question, model_gpt,
            n, ",".join(sorted(brand_variants))
        )
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT: simulation({question[:30]}...)")
            return SimResult(cache_hit=True, **cached)
    else:
        cache_key = None

    sim_prompt = f"질문: {question}\n\n답변:"

    def _gpt_call(q: str) -> str:
        return call_gpt(
            client_gpt, q,
            max_tokens=180,
            model=model_gpt,
            temperature=0.6,
            tracker=tracker,
        )

    def _gem_call(q: str) -> str:
        return call_gemini(
            client_gemini, q,
            max_tokens=180,
            temperature=0.6,
            tracker=tracker,
        )

    gpt_hits, gpt_samples, gpt_n = 0, [], 0
    gem_hits, gem_samples, gem_n = 0, [], 0
    gpt_ran = False
    gem_ran = False

    # BUG FIX: 타임아웃 상한 현실화 — 최대 120초, 최소 60초
    timeout = min(120, max(60, n * 2))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if client_gpt:
            futures["gpt"] = ex.submit(
                _adaptive_batch, _gpt_call, sim_prompt, brand_variants, n
            )
        if client_gemini:
            futures["gem"] = ex.submit(
                _adaptive_batch, _gem_call, sim_prompt, brand_variants, n,
                0.05, True  # count_mention=True: 브랜드 언급도 집계
            )

        gpt_comp = {}
        gem_comp = {}
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
        all_comps = set(list(gpt_comp.keys()) + list(gem_comp.keys()))
        merged_comp = {}
        for c in all_comps:
            gm = gpt_comp.get(c, {"mentions":0,"urls":[]})
            gn = gem_comp.get(c, {"mentions":0,"urls":[]})
            merged_comp[c] = {
                "mentions": gm["mentions"] + gn["mentions"],
                "urls": list(set(gm["urls"] + gn["urls"]))
            }

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
        gpt_hits=gpt_hits   if gpt_ran else None,
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
    # BUG FIX: None 잔존 방지용 기본 SimResult 팩토리
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

    max_workers = min(len(questions), 5)
    # BUG FIX: 전체 타임아웃도 현실화
    collect_timeout = min(300, max(90, n * 3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_sim_one, i, q): i for i, q in enumerate(questions)}
        for fut in concurrent.futures.as_completed(futs):
            with CaptureError("sim_future_collect", log_level="warning"):
                fut.result(timeout=collect_timeout)

    # BUG FIX: None이 남아있는 슬롯 교체 (스레드 예외로 results[idx] 미설정된 경우)
    results = [r if r is not None else _empty_result() for r in results]

    return results
