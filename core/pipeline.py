"""
pipeline.py — Crawler → Biz → Question 통합 오케스트레이터

수정 사항 (버그픽스):
1. ThreadPoolExecutor 누수 — try/finally로 항상 shutdown 보장
2. 크롤 타임아웃 시 shutdown(wait=False) — block 없이 즉시 반환
3. 크롤 실패 시 1회 재시도 로직 추가
4. comp_future None 체크 추가
5. 캐시 히트 시 _do_competitors()가 biz_info 없이 실행되는 버그 수정
6. CrawlResult 역직렬화 누락 필드 방어 처리 (get + 기본값)
7. render_debug_panel 마크다운 이스케이프 수정
8. question 캐시 히트 후 재진입 방어
"""

from __future__ import annotations

import re
import json
import time
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from core.logger import get_logger, CaptureError
from core.cache import get_cache
from core.crawler import crawl, crawl_search, CrawlResult
from core.citation import (
    build_brand_variants, detect_citation, CitationResult, wilson_ci
)
from core.ai_client import call_gpt, call_gemini, CostTracker
from core.biz_analysis import (
    BusinessInfo, Competitor,
    _CATEGORY_HINTS, _QUESTION_SYSTEM,
    discover_competitors,
)
from core.industry_classifier import _BIZ_SYSTEM, _build_classify_prompt
from core.schemas import _BAD_INDUSTRIES

def _build_biz_prompt(domain: str, crawl_result, search_ctx: str) -> str:
    clean_text = (crawl_result.body_text if crawl_result and crawl_result.body_text else "")
    return _build_classify_prompt(domain, clean_text, search_ctx, "")

logger = get_logger("pipeline")


# ─────────────────────────────────────────────
# 파이프라인 상태
# ─────────────────────────────────────────────

@dataclass
class PipelineState:
    url: str
    domain: str
    crawl_result: Optional[CrawlResult] = None
    search_ctx: str = ""
    biz_info: Optional[BusinessInfo] = None
    competitors: list[Competitor] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    spot_check: dict = field(default_factory=dict)
    debug_log: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def log(self, stage: str, data: dict):
        self.debug_log.append({"stage": stage, "ts": time.time(), **data})

    def record_time(self, stage: str, elapsed: float):
        self.timing[stage] = round(elapsed, 2)


# ─────────────────────────────────────────────
# Content Filter
# ─────────────────────────────────────────────

_WEAK_PATTERNS = [
    r"(?:는|이|가|을|를)\s*무엇인가요",
    r"소개해\s*주세요",
    r"설명해\s*주세요",
    r"어디\s*(?:있|위치)",
    r"역사.*어떻게",
    r"what\s+is\b",
    r"tell\s+me\s+about",
    r"introduce\b",
]

_STRONG_PATTERNS = [
    r"\d+\s*%",
    r"(?:비교|대비|vs\.?|versus)",
    r"(?:ROAS|CPA|CTR|ROI|CPM|CPV)",
    r"(?:비용|수수료|가격|요금|단가)",
    r"(?:사례|레퍼런스|후기|리뷰)",
    r"(?:경쟁사|대안|차이점|장단점)",
    r"(?:how\s+much|compare|versus|case\s+study|pricing)",
]

def content_filter(question: str, brand_name: str) -> dict:
    score = 50
    flags = []

    for pat in _WEAK_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            score -= 20
            flags.append(f"weak_pattern: {pat}")
            break

    for pat in _STRONG_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            score += 15
            flags.append(f"strong_pattern: {pat}")

    if brand_name and brand_name.lower() in question.lower():
        score -= 15
        flags.append("brand_direct_included")
    else:
        score += 10
        flags.append("brand_not_included_good")

    if len(question) < 20:
        score -= 20
        flags.append("too_short")
    elif len(question) > 150:
        score -= 5
        flags.append("too_long")

    if not question.strip().endswith("?"):
        score -= 10
        flags.append("no_question_mark")

    score = max(0, min(100, score))
    passed = score >= 40
    reason = "pass" if passed else f"low_score({score}): " + ", ".join(flags[:2])
    return {"score": score, "pass": passed, "reason": reason, "flags": flags}


# ─────────────────────────────────────────────
# Citation Spot-Check
# ─────────────────────────────────────────────

def citation_spot_check(
    client_gpt, client_gemini,
    brand_variants: list[str],
    question: str,
    model_gpt: str,
    n_samples: int = 5,
    tracker: Optional[CostTracker] = None,
) -> dict:
    prompt = f"질문: {question}\n\n답변:"
    samples = []
    hits = 0

    for i in range(n_samples):
        resp = ""
        with CaptureError(f"spot_gpt_{i}", log_level="debug"):
            if client_gpt:
                resp = call_gpt(client_gpt, prompt, max_tokens=200,
                                model=model_gpt, temperature=0.6, tracker=tracker)
            elif client_gemini:
                resp = call_gemini(client_gemini, prompt, max_tokens=200,
                                   temperature=0.6, tracker=tracker)
        if not resp:
            continue

        result: CitationResult = detect_citation(resp, brand_variants)
        samples.append({
            "response": resp[:300],
            "cited": result.cited,
            "confidence": result.confidence,
            "pattern_type": result.matches[0].pattern_type if result.matches else None,
            "is_negative": result.matches[0].is_negative if result.matches else False,
        })
        if result.cited:
            hits += 1

    hit_rate = hits / max(len(samples), 1) * 100

    low_conf_hits = sum(
        1 for s in samples
        if s["cited"] and s.get("confidence", 0) < 0.45
    )
    fp_ratio = low_conf_hits / max(hits, 1) if hits > 0 else 0

    if fp_ratio > 0.5:
        fp_risk = "high"
    elif fp_ratio > 0.2:
        fp_risk = "medium"
    else:
        fp_risk = "low"

    suspect  = [v for v in brand_variants if len(v) <= 3 or
                v.lower() in {"app", "net", "web", "the", "inc", "co"}]
    refined  = [v for v in brand_variants if v not in suspect] or brand_variants

    return {
        "hit_rate": round(hit_rate, 1),
        "false_positive_risk": fp_risk,
        "suspect_variants": suspect,
        "sample_responses": samples,
        "refined_variants": refined,
        "n_samples": len(samples),
    }


# ─────────────────────────────────────────────
# Biz 분석 (CrawlResult 직접 수신)
# ─────────────────────────────────────────────

def analyze_business_from_crawl(
    client_gpt, client_gemini,
    state: PipelineState,
    model_gpt: str,
    confirmed_industry: str = "",
) -> BusinessInfo:
    crawl_result = state.crawl_result
    domain       = state.domain
    search_ctx   = state.search_ctx
    stem         = domain.split(".")[0]
    t0           = time.time()

    body_len     = len(crawl_result.body_text) if crawl_result else 0
    data_quality = "rich" if body_len > 1000 else ("sparse" if body_len > 200 else "empty")

    state.log("biz_input", {
        "domain": domain,
        "crawl_tier": crawl_result.tier_used if crawl_result else 0,
        "body_len": body_len,
        "data_quality": data_quality,
        "has_search_ctx": bool(search_ctx),
    })

    prompt     = _build_biz_prompt(domain, crawl_result or CrawlResult(url=domain), search_ctx)
    result_str = ""

    with CaptureError("biz_ai", log_level="warning") as ctx:
        if client_gpt:
            result_str = call_gpt(client_gpt, prompt, system=_BIZ_SYSTEM,
                                  max_tokens=500, model=model_gpt, temperature=0.15)
        elif client_gemini:
            result_str = call_gemini(client_gemini, prompt, max_tokens=500, temperature=0.15)

    if not ctx.ok:
        state.errors.append(f"biz_ai: {ctx.error}")

    biz_dict = {}
    with CaptureError("biz_parse", log_level="warning"):
        m = re.search(r'\{.*\}', result_str, re.DOTALL)
        if m:
            biz_dict = json.loads(m.group())

    if biz_dict:
        industry = biz_dict.get("industry", "")
        is_vague = any(bad in industry.lower() for bad in _BAD_INDUSTRIES)
        if is_vague and ((crawl_result and crawl_result.body_text) or search_ctx):
            retry_ctx = (
                f"사이트 본문: {crawl_result.body_text[:800]}\n"
                f"검색 보완: {search_ctx[:500]}"
                if crawl_result else search_ctx[:1000]
            )
            retry_p = f"""도메인 "{domain}"의 업종을 아래 실제 콘텐츠로 정확히 분류하세요.

{retry_ctx}

키워드가 있으면 분류 기준:
- 광고/마케팅 키워드 → "퍼포먼스 마케팅 광고대행사" 류
- 쇼핑/상품 키워드 → "온라인 쇼핑몰" 류
- SaaS/B2B 키워드 → "B2B SaaS [기능]" 류
- 교육 키워드 → "온라인 교육 플랫폼" 류

JSON만 출력: {{"industry": "구체적 업종명"}}"""

            with CaptureError("biz_industry_retry", log_level="warning"):
                r2 = (
                    call_gpt(client_gpt, retry_p, max_tokens=80, model=model_gpt, temperature=0.1)
                    if client_gpt else
                    call_gemini(client_gemini, retry_p, max_tokens=80, temperature=0.1)
                )
                m2 = re.search(r'\{.*\}', r2, re.DOTALL)
                if m2:
                    ni = json.loads(m2.group()).get("industry", "")
                    if ni and not any(bad in ni.lower() for bad in _BAD_INDUSTRIES):
                        biz_dict["industry"] = ni
                        state.log("biz_industry_retry", {"original": industry, "corrected": ni})

    if not biz_dict:
        state.errors.append("biz_parse: JSON 파싱 실패, 폴백 사용")
        biz_dict = {
            "brand_name":       stem.upper(),
            "industry":         f"{stem} 서비스",
            "industry_category":"기타",
            "core_product":     "서비스",
            "target_audience":  "잠재 고객",
            "key_services":     [],
            "confidence":       "low",
        }

    biz_dict["crawl_tier"] = crawl_result.tier_used if crawl_result else 0

    if confirmed_industry.strip():
        biz_dict["industry"]    = confirmed_industry.strip()
        biz_dict["confidence"]  = "high"
        state.log("biz_user_override", {"industry": confirmed_industry})

    biz = BusinessInfo.from_dict(biz_dict)
    state.record_time("biz_analysis", time.time() - t0)
    state.log("biz_output", biz.to_dict())
    return biz


# ─────────────────────────────────────────────
# 질문 생성 (crawl + biz 모두 참조)
# ─────────────────────────────────────────────

def generate_questions_from_state(
    client_gpt, client_gemini,
    state: PipelineState,
    model_gpt: str,
    engine: str = "GPT",
) -> list[str]:
    biz       = state.biz_info
    crawl_res = state.crawl_result

    if biz is None:
        state.errors.append("question_gen: biz_info 없음")
        return []

    t0 = time.time()

    site_keywords = ""
    if crawl_res and crawl_res.body_text:
        body    = crawl_res.body_text[:2000]
        ko_words = re.findall(r'[가-힣]{2,}', body)
        top_ko   = [w for w, _ in Counter(ko_words).most_common(10) if len(w) >= 2]
        en_words = re.findall(r'[A-Z][a-zA-Z]{3,}', body)
        top_en   = list(dict.fromkeys(en_words))[:5]
        site_keywords = ", ".join(top_ko[:8] + top_en[:3])

    category_hint = _CATEGORY_HINTS.get(
        biz.industry_category,
        f"{biz.industry} 분야에서 {biz.target_audience}가 실제로 고민하는 핵심 질문"
    )
    services_str = ", ".join(biz.key_services) if biz.key_services else biz.core_product

    prompt = f"""당신은 {biz.industry} 분야 10년 경력 마케팅 전략가이자 GEO 전문가입니다.

[분석 대상 — 실제 사이트 분석 결과]
- 업종: {biz.industry} ({biz.industry_category})
- 핵심 서비스: {services_str}
- 주요 타겟: {biz.target_audience}
- 사이트 실제 키워드: {site_keywords if site_keywords else "(크롤 데이터 없음)"}
- 크롤 품질: Tier{biz.crawl_tier} / 신뢰도 {biz.confidence}

[핵심 목표]
{biz.target_audience}가 ChatGPT, Gemini, 네이버 AI 등에게 실제로 물어볼 법한 질문을 만드세요.
이 질문에 AI가 답할 때 {biz.brand_name} 사이트가 자연스럽게 인용될 수 있어야 합니다.

[질문 방향]
{category_hint}

[생성 규칙 — 엄격 준수]
1. 브랜드명({biz.brand_name}) 절대 포함 금지 — 실제 사용자는 브랜드를 모르고 검색함
2. 위 "사이트 실제 키워드"를 질문 소재로 적극 활용 (실제 사이트 서비스 반영)
3. 구매 결정 5단계(인지→비교→신뢰→가격→전환)를 각각 다룰 것
4. {biz.industry} 전문 용어·지표·관행 적극 활용
5. "~는 무엇인가요?", "~를 소개해주세요" 금지
6. 구체적 수치·비교·상황·지역 포함 필수

[좋은 예시]
- "네이버 공식 광고대행사 추천해줘"
- "중소기업 구글 광고 맡길 대행사 어디가 좋아?"
- "퍼포먼스 마케팅 잘하는 광고대행사 ROAS 비교"

[품질 기준]
✅ 브랜드명 없이 업종/서비스 키워드만으로 구성
✅ 비교/수치/사례/비용/지역 중 하나 이상 포함
✅ 실제 구매·도입 맥락에서 나올 법한 자연스러운 질문
❌ 브랜드명, 회사명, 도메인 주소 포함
❌ 단순 정보 탐색형 (역사, 소개, 설명 등)

번호·라벨 없이 질문 5개만 출력. 한 줄에 하나. 물음표(?)로 종결.
절대 금지: 마크다운 기호(_arrowRight_, *, **, →, ▶, :emoji:), 이모지, 특수문자 사용 금지. 순수 텍스트만 출력."""

    result_str = ""
    with CaptureError("question_gen", log_level="warning") as ctx:
        if engine == "GPT" and client_gpt:
            result_str = call_gpt(client_gpt, prompt, system=_QUESTION_SYSTEM,
                                  max_tokens=600, model=model_gpt, temperature=0.85)
        elif client_gemini:
            result_str = call_gemini(client_gemini, prompt, max_tokens=600, temperature=0.85)
        elif client_gpt:
            result_str = call_gpt(client_gpt, prompt, system=_QUESTION_SYSTEM,
                                  max_tokens=600, model=model_gpt, temperature=0.85)

    if not ctx.ok:
        state.errors.append(f"question_gen: {ctx.error}")

    lines = [ln.strip() for ln in result_str.split("\n") if ln.strip()]
    raw_questions = []
    for ln in lines:
        clean = ln
        clean = re.sub(r'_[a-zA-Z]+_', '', clean)
        clean = re.sub(r':[a-z_]+:', '', clean)
        clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', clean)
        clean = re.sub(r'^[\d]+[.)]\s*', '', clean)
        clean = re.sub(r'^[-•*→▶►•◦‣⁃]\s*', '', clean)
        clean = re.sub(r'^\[.*?\]\s*', '', clean)
        clean = re.sub(r'[☀-➿]', '', clean)
        clean = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if len(clean) > 10 and not clean.startswith("참고") and not clean.startswith("주의"):
            if not clean.endswith("?"):
                clean += "?"
            raw_questions.append(clean)

    filtered       = []
    filter_results = []
    for q in raw_questions:
        fr = content_filter(q, biz.brand_name)
        filter_results.append({"question": q, **fr})
        if fr["pass"]:
            filtered.append(q)
        else:
            state.log("question_filtered", {"question": q, "reason": fr["reason"]})

    state.log("question_gen", {
        "raw_count":          len(raw_questions),
        "filtered_count":     len(filtered),
        "filter_results":     filter_results,
        "site_keywords_used": site_keywords,
    })

    questions = filtered[:5]

    if len(questions) < 3:
        for q in raw_questions:
            if q not in questions:
                questions.append(q)
            if len(questions) >= 5:
                break

    if len(questions) < 3:
        is_ad = "광고" in biz.industry or "마케팅" in biz.industry
        state.errors.append("question_gen: 폴백 질문 사용")
        questions = (
            [
                f"{biz.industry} 추천해줘 ROAS 높은 곳?",
                f"{biz.industry} 계약 전 확인해야 할 대행수수료 구조는?",
                f"중소기업 {biz.industry} 맡길 때 업종별 성공 사례 알려줘",
                f"광고 직접 운영 vs {biz.industry} 위탁 — 비용 차이 비교",
                f"{biz.industry} 잘하는 곳 어떻게 비교하나요?",
            ] if is_ad else [
                f"{biz.industry} 서비스 비교할 때 핵심 기준은?",
                f"{biz.target_audience} 에게 맞는 {biz.industry} 추천해줘",
                f"{biz.industry} 비용 구조 업계 평균 어느 정도야?",
                f"{biz.industry} 실제 사용 후기 어디서 봐?",
                f"{biz.industry} 선택할 때 가장 중요한 게 뭐야?",
            ]
        )

    state.record_time("question_gen", time.time() - t0)
    return questions[:5]


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def run_pipeline(
    client_gpt, client_gemini,
    url: str,
    model_gpt: str,
    confirmed_industry: str = "",
    confirmed_brand: str = "",
    q_engine: str = "GPT",
    market_scope: str = "국내 (대한민국)",
    n_competitors: int = 3,
    tracker: Optional[CostTracker] = None,
    use_cache: bool = True,
    debug: bool = False,
    status_callback=None,
) -> PipelineState:

    from urllib.parse import urlparse

    try:
        p      = urlparse(url if url.startswith("http") else "https://" + url)
        domain = p.netloc.replace("www.", "")
    except Exception:
        domain = url

    state = PipelineState(url=url, domain=domain)
    cache = get_cache()

    def _cb(stage: str, msg: str):
        if status_callback:
            status_callback(stage, msg)
        if debug:
            logger.info(f"[{stage}] {msg}")

    # ── Step 1: 크롤링 + 검색 보완 ──
    _cb("crawl", f"{domain} 크롤링 중 (3-tier fallback)...")
    t0 = time.time()

    cached_crawl = None
    if use_cache:
        key          = cache.make_key("pipeline_crawl", url)
        cached_crawl = cache.get(key)
        if cached_crawl:
            cr_data = cached_crawl["crawl"]
            state.crawl_result = CrawlResult(
                url          = cr_data.get("url", url),
                title        = cr_data.get("title", ""),
                description  = cr_data.get("description", ""),
                body_text    = cr_data.get("body_text", ""),
                html_snippet = cr_data.get("html_snippet", ""),
                tier_used    = cr_data.get("tier_used", 0),
                ok           = cr_data.get("ok", False),
                error        = cr_data.get("error", ""),
            )
            state.search_ctx = cached_crawl.get("search_ctx", "")
            _cb("crawl", f"캐시 히트 (Tier{state.crawl_result.tier_used})")
            state.record_time("crawl", 0)

    if state.crawl_result is None:
        stem = domain.split(".")[0]

        def _do_crawl():
            return crawl(url, use_cache=False)

        def _do_search():
            ctx = crawl_search(f"{stem} 서비스 업종 소개")
            if not ctx:
                ctx = crawl_search(f"{stem} company overview")
            return ctx or ""

        # FIX: try/finally + shutdown(wait=False) — timeout 후 block 없이 반환
        _crawl_ex = cf.ThreadPoolExecutor(max_workers=2)
        try:
            f_crawl  = _crawl_ex.submit(_do_crawl)
            f_search = _crawl_ex.submit(_do_search)

            with CaptureError("crawl_future", log_level="warning") as ctx:
                state.crawl_result = f_crawl.result(timeout=25)
            if not ctx.ok:
                state.errors.append(f"crawl: {ctx.error}")
                state.crawl_result = CrawlResult(url=url, ok=False, tier_used=0)
                # FIX: 크롤 실패 시 1회 재시도 (검색 보완 먼저 수집)
                _cb("crawl", "크롤 실패, 재시도 중...")
                with CaptureError("crawl_retry", log_level="warning"):
                    state.crawl_result = crawl(url, use_cache=False)

            with CaptureError("search_future", log_level="warning"):
                state.search_ctx = f_search.result(timeout=20)
        finally:
            _crawl_ex.shutdown(wait=False)

        if use_cache and state.crawl_result and state.crawl_result.ok:
            cache.set(cache.make_key("pipeline_crawl", url), {
                "crawl": {
                    "url":          state.crawl_result.url,
                    "title":        state.crawl_result.title,
                    "description":  state.crawl_result.description,
                    "body_text":    state.crawl_result.body_text,
                    "html_snippet": getattr(state.crawl_result, "html_snippet", ""),
                    "tier_used":    state.crawl_result.tier_used,
                    "ok":           state.crawl_result.ok,
                    "error":        state.crawl_result.error,
                },
                "search_ctx": state.search_ctx,
            }, namespace="crawl")

    if state.crawl_result is None:
        state.crawl_result = CrawlResult(url=url, ok=False, tier_used=0)

    state.record_time("crawl", time.time() - t0)
    state.log("crawl", {
        "tier":           state.crawl_result.tier_used,
        "ok":             state.crawl_result.ok,
        "body_len":       len(state.crawl_result.body_text),
        "title":          state.crawl_result.title[:80],
        "search_ctx_len": len(state.search_ctx),
    })
    _cb("crawl", f"완료 (Tier{state.crawl_result.tier_used}, 본문 {len(state.crawl_result.body_text)}자)")

    # ── Step 2: Biz 분석 ──
    _cb("biz", "업종 AI 분석 중...")
    t0 = time.time()

    biz_cache_key = cache.make_key("pipeline_biz", url, confirmed_industry)
    if use_cache:
        cached_biz = cache.get(biz_cache_key)
        if cached_biz:
            state.biz_info = BusinessInfo.from_dict(cached_biz)
            _cb("biz", f"캐시 히트: {state.biz_info.brand_name} | {state.biz_info.industry}")

    if state.biz_info is None:
        state.biz_info = analyze_business_from_crawl(
            client_gpt, client_gemini, state,
            model_gpt=model_gpt,
            confirmed_industry=confirmed_industry,
        )
        if use_cache:
            cache.set(biz_cache_key, state.biz_info.to_dict(), namespace="biz")

    state.record_time("biz_analysis", time.time() - t0)

    if confirmed_brand.strip() and state.biz_info:
        state.biz_info.brand_name = confirmed_brand.strip()
        state.log("brand_override", {"brand_name": confirmed_brand.strip()})

    _cb("biz", f"완료: {state.biz_info.brand_name} | {state.biz_info.industry} | {state.biz_info.confidence}")

    # ── Step 3: 경쟁사 + 질문 병렬 실행 ──
    _cb("competitors", f"[{market_scope}] 경쟁사 분석 중...")
    _cb("questions",   "크롤 데이터 기반 타겟 질문 도출 중...")

    t0          = time.time()
    comp_future = None
    executor    = cf.ThreadPoolExecutor(max_workers=2)

    try:
        def _do_competitors():
            return discover_competitors(
                client_gpt, client_gemini,
                state.biz_info, url,
                market_scope=market_scope,
                model_gpt=model_gpt,
                n_competitors=n_competitors,
                use_cache=use_cache,
            )

        comp_future = executor.submit(_do_competitors)

        q_cache_key = cache.make_key(
            "pipeline_questions", url, state.biz_info.industry, q_engine
        )
        if use_cache:
            cached_q = cache.get(q_cache_key)
            if cached_q:
                state.questions = cached_q
                _cb("questions", f"캐시 히트: {len(state.questions)}개")

        if not state.questions:
            state.questions = generate_questions_from_state(
                client_gpt, client_gemini, state,
                model_gpt=model_gpt, engine=q_engine,
            )
            if use_cache and state.questions:
                cache.set(q_cache_key, state.questions, namespace="biz")

        state.record_time("question_gen", time.time() - t0)
        _cb("questions", f"완료: {len(state.questions)}개 질문 (content filter 통과)")

        if comp_future is not None:
            with CaptureError("comp_collect", log_level="warning") as ctx:
                state.competitors = comp_future.result(timeout=60)
            if not ctx.ok:
                state.errors.append(f"competitors: {ctx.error}")
                state.competitors = []
        _cb("competitors", f"완료: {len(state.competitors)}개")

    finally:
        executor.shutdown(wait=False)

    # ── Step 4: Citation Spot-Check ──
    if state.questions and (client_gpt or client_gemini):
        _cb("spot_check", "Citation 정확도 spot-check 중 (5회 샘플)...")
        t0 = time.time()

        brand_variants = build_brand_variants(url, state.biz_info.to_dict())
        first_q        = state.questions[0]

        spot = citation_spot_check(
            client_gpt, client_gemini,
            brand_variants=brand_variants,
            question=first_q,
            model_gpt=model_gpt,
            n_samples=5,
            tracker=tracker,
        )
        state.spot_check = spot
        state.record_time("spot_check", time.time() - t0)
        state.log("spot_check", spot)
        _cb("spot_check",
            f"완료 | 초기 점유율: {spot['hit_rate']}% | FP 리스크: {spot['false_positive_risk']}")

    return state


# ─────────────────────────────────────────────
# Debug 패널 렌더링
# ─────────────────────────────────────────────

def render_debug_panel(state: PipelineState):
    import streamlit as st
    import pandas as pd

    st.markdown("---")
    st.markdown("### 🔬 Debug 패널")

    if state.timing:
        cols = st.columns(len(state.timing))
        for i, (k, v) in enumerate(state.timing.items()):
            cols[i].metric(k, f"{v}s")

    if state.errors:
        st.error("파이프라인 에러:\n" + "\n".join(f"• {e}" for e in state.errors))

    with st.expander("🌐 Crawl 결과", expanded=False):
        if state.crawl_result:
            cr = state.crawl_result
            st.json({
                "tier":         cr.tier_used,
                "ok":           cr.ok,
                "title":        cr.title,
                "description":  cr.description[:200],
                "body_len":     len(cr.body_text),
                "body_preview": cr.body_text[:500],
            })
        else:
            st.warning("크롤 결과 없음")
        if state.search_ctx:
            st.text_area("검색 보완 컨텍스트", value=state.search_ctx[:1000], height=150)

    with st.expander("🏢 Biz 분석 결과", expanded=False):
        if state.biz_info:
            st.json(state.biz_info.to_dict())
        else:
            st.warning("Biz 분석 결과 없음")

    with st.expander("📝 질문 생성 + Content Filter", expanded=False):
        q_log = [l for l in state.debug_log if l["stage"] in ("question_gen", "question_filtered")]
        if q_log:
            for entry in q_log:
                if entry["stage"] == "question_gen" and "filter_results" in entry:
                    rows = []
                    for fr in entry["filter_results"]:
                        rows.append({
                            "질문": fr["question"][:60],
                            "점수": fr["score"],
                            "통과": "✅" if fr["pass"] else "❌",
                            "사유": fr["reason"][:50],
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption(f"사이트 키워드: {entry.get('site_keywords_used', '없음')}")
                elif entry["stage"] == "question_filtered":
                    st.caption(f"❌ 필터링됨: {entry.get('question','')[:50]} — {entry.get('reason','')}")
        else:
            st.info("질문 생성 로그 없음")

    with st.expander("🎯 Citation Spot-Check", expanded=False):
        if state.spot_check:
            sp = state.spot_check
            c1, c2, c3 = st.columns(3)
            c1.metric("초기 점유율 추정", f"{sp['hit_rate']}%")
            c2.metric("FP 리스크",        sp["false_positive_risk"])
            c3.metric("샘플 수",          sp["n_samples"])

            if sp.get("suspect_variants"):
                st.warning(f"⚠️ 오탐 의심 변형: {sp['suspect_variants']}")
            if sp.get("refined_variants"):
                st.info(f"✅ 정제된 변형 ({len(sp['refined_variants'])}개): {sp['refined_variants']}")

            for i, s in enumerate(sp.get("sample_responses", [])[:3]):
                color = "✅" if s["cited"] else "❌"
                conf  = s.get("confidence", 0)
                st.markdown(
                    f"**샘플 {i+1}** {color} "
                    f"confidence={conf:.2f} "
                    f"pattern={s.get('pattern_type','—')}"
                )
                st.caption(s["response"][:200])
        else:
            st.info("Spot-check 데이터 없음")

    with st.expander("📋 전체 파이프라인 로그", expanded=False):
        for entry in state.debug_log:
            stage = entry.get("stage", "?")
            ts    = entry.get("ts", 0)
            rest  = {k: v for k, v in entry.items() if k not in ("stage", "ts")}
            st.markdown(f"**[{stage}]** `{ts:.2f}`")
            st.json(rest)
