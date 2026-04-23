"""
strategy_analyzer.py — 전략 분석 전담 모듈

경쟁사 분석 · 인용 실패 진단 · 블루오션 키워드 · GEO 가이드 4개 병렬 실행.

수정 사항 (버그픽스):
1. from .prompts import ... 제거 — prompts.py 없으면 ImportError 앱 크래시
   → 프롬프트 템플릿을 이 파일 내부에 직접 정의
2. _diagnosis/_keywords/_geo 에서 CaptureError 실패 시 r 미정의 → UnboundLocalError
   → r = "" 초기화 후 블록 진입
3. cache.get() 히트 체크 `if cached:` → `if cached is not None:`
4. _safe_result timeout 파라미터 — with 블록 종료 후 이미 완료되므로 로그만 명확화
"""

from __future__ import annotations

import re
import json
import concurrent.futures
from typing import Optional
from urllib.parse import urlparse

from .logger import get_logger, CaptureError
from .cache import get_cache
from .ai_client import call_gpt, call_gemini
from .schemas import BusinessInfo
from .crawler import crawl

logger = get_logger("strategy_analyzer")

_STRATEGY_VERSION = "v4"

# ─────────────────────────────────────────────
# 프롬프트 상수 (prompts.py 의존성 제거)
# ─────────────────────────────────────────────

STRATEGY_SYSTEM = (
    "당신은 디지털 마케팅 전략 컨설턴트입니다. "
    "모든 답변은 완성된 문장으로, 중간에 끊기지 않게 서술하세요."
)

STRATEGY_COMP_TMPL = """질문: "{question}"

이 질문에 답변할 때 AI가 자주 인용할 상위 10개 브랜드를 인용 가능성 순으로 나열하세요.
- 분석 대상: {brand_name}
- {scope_inst}
- {domain}도 적절한 순위에 포함
- 반드시 실제 존재하는 브랜드와 도메인만 사용. 가상 도메인 절대 금지.

JSON 배열만 출력 (다른 텍스트 없이):
[{{"rank":1,"domain":"실제도메인.com","brand_name":"실제브랜드명","reason":"이유 20자 이내","position":"업계1위|신흥강자|틈새전문 중 택1"}}]"""

STRATEGY_DIAG_TMPL = """아래는 {domain} ({brand_name})의 실제 사이트 내용입니다.

[사이트 내용]
{site_content}

위 사이트가 "{question}" 같은 질문에서 AI 인용 점유율이 낮은 원인 3가지를 분석하세요.
사이트 내용을 직접 분석해서 구체적 문제점을 지적하세요. 각 항목 60자 이내. 번호 없이 한 줄씩:"""

STRATEGY_KW_TMPL = """아래는 {domain} ({brand_name})의 실제 사이트 내용입니다.

[사이트 내용]
{site_content}

위 사이트 내용을 바탕으로, ChatGPT·Gemini·Perplexity가 자주 인용할 블루오션 키워드 5개를 추천하세요.
조건: 이 사이트의 실제 서비스/강점 반영 · 틈새 키워드 · {scope_inst} · 한국어만
키워드만 한 줄에 하나씩, 5개만 출력:"""

STRATEGY_GEO_TMPL = """아래는 {domain} ({brand_name})의 실제 사이트 내용입니다.

[사이트 내용]
{site_content}

위 사이트가 "{question}" 같은 질문에서 AI에 더 자주 인용되도록 개선 방안 3가지를 작성하세요.
각 항목: 1.무엇을 바꾸는지 2.어떻게 바꾸는지(실제 문구/구조) 3.왜 AI 인용에 도움되는지
형식: 번호. **제목**\n내용 (3줄 이내)
출력 언어: 한국어"""


def run_strategy_analysis(
    client_gpt,
    client_gemini,
    # app.py 새 호출 방식 (biz_info/competitors/sim_results/questions 키워드)
    biz_info:    Optional[BusinessInfo] = None,
    competitors: list = None,
    sim_results: list = None,
    questions:   list = None,
    tracker=None,
    # 레거시 파라미터 (하위 호환)
    question:    str = "",
    target_url:  str = "",
    model_gpt:   str = "gpt-4o-mini",
    market_scope: str = "글로벌",
    use_cache:   bool = True,
) -> dict:
    """
    경쟁사·진단·키워드·GEO 가이드 4개를 병렬로 생성해 반환.
    app.py의 새 호출 방식(keyword args)과 레거시 방식 모두 지원.
    """
    competitors = competitors or []
    sim_results = sim_results or []
    questions   = questions   or []

    # target_url / domain 결정
    _url = target_url or (biz_info.brand_name if biz_info else "")
    try:
        p      = urlparse(_url if _url.startswith("http") else "https://" + _url)
        domain = p.netloc.replace("www.", "") or _url
    except Exception:
        domain = _url

    # 대표 질문 결정
    rep_question = question or (questions[0] if questions else "AI 인용 점유율 분석")

    cache     = get_cache()
    cache_key = cache.make_key(
        "strategy", _STRATEGY_VERSION,
        domain, rep_question[:50], market_scope, model_gpt,
    )

    # BUG FIX: `if cached:` → `if cached is not None:`
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache HIT: strategy({rep_question[:30]}...)")
            return cached

    biz = biz_info or BusinessInfo(
        brand_name=domain, industry="서비스", industry_category="기타",
        core_product="서비스", target_audience="고객",
    )

    # URL 크롤링 — 진단/키워드/GEO에 실제 사이트 내용 활용
    site_content = ""
    if target_url:
        with CaptureError("strategy_crawl", log_level="warning"):
            crawl_result = crawl(target_url, use_cache=True)
            if crawl_result.ok and crawl_result.body_text:
                site_content = crawl_result.summary(max_len=1500)
    if not site_content:
        site_content = f"{domain} ({biz.brand_name}) 사이트 — 크롤 데이터 없음"

    scope_inst = (
        "반드시 대한민국에서 서비스하는 국내 기업만 포함하세요."
        if "국내" in market_scope
        else "국내외 글로벌 기업을 모두 포함하세요."
    )

    # 시뮬레이션 결과 컨텍스트
    sim_ctx = ""
    if sim_results and questions:
        lines = []
        for q, r in zip(questions[:3], sim_results[:3]):
            avg = getattr(r, "avg_rate", None) or (r.get("avg_rate") if isinstance(r, dict) else None)
            if avg is not None:
                lines.append(f"- Q: {q[:40]} → 평균 점유율 {avg:.1f}%")
        if lines:
            sim_ctx = "\n[시뮬레이션 결과]\n" + "\n".join(lines)

    # ── 4개 서브태스크 ──────────────────────────

    def _competitors_fn() -> list:
        # 이미 competitors가 있으면 재사용
        if competitors:
            result = []
            for c in competitors[:10]:
                if isinstance(c, dict):
                    result.append(c)
                elif hasattr(c, "to_dict"):
                    result.append(c.to_dict())
                elif hasattr(c, "__dict__"):
                    result.append(vars(c))
            return result

        prompt = STRATEGY_COMP_TMPL.format(
            question=rep_question,
            brand_name=biz.brand_name,
            industry=biz.industry,
            scope_inst=scope_inst,
            domain=domain,
        )
        # BUG FIX: r = "" 초기화
        raw = ""
        with CaptureError("strategy_comp", log_level="warning"):
            raw = (
                call_gpt(client_gpt, prompt, system=STRATEGY_SYSTEM,
                         max_tokens=1200, model=model_gpt, temperature=0.3)
                if client_gpt else
                call_gemini(client_gemini, prompt, max_tokens=1200, temperature=0.3)
            )
        if raw:
            with CaptureError("strategy_comp_parse", log_level="warning"):
                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    real = [
                        c for c in parsed
                        if c.get("domain") and
                        not re.match(
                            r'^(c\d+|competitor\d*|example|test|dummy)\.',
                            str(c.get("domain", ""))
                        )
                    ]
                    if real:
                        return real
        return []

    def _diagnosis() -> list[str]:
        prompt = STRATEGY_DIAG_TMPL.format(
            domain=domain,
            brand_name=biz.brand_name,
            site_content=site_content,
            question=rep_question,
        ) + sim_ctx

        # BUG FIX: r = "" 초기화 → CaptureError 실패 시 UnboundLocalError 방지
        r = ""
        with CaptureError("strategy_diag", log_level="warning"):
            r = (
                call_gpt(client_gpt, prompt, system=STRATEGY_SYSTEM,
                         max_tokens=600, model=model_gpt, temperature=0.4)
                if client_gpt else
                call_gemini(client_gemini, prompt, max_tokens=600, temperature=0.4)
            )
        items = [d.strip().lstrip("•-*") for d in r.split("\n") if d.strip()][:3]
        return items or ["데이터 부족으로 분석 불가"]

    def _keywords() -> list[str]:
        prompt = STRATEGY_KW_TMPL.format(
            domain=domain,
            brand_name=biz.brand_name,
            site_content=site_content,
            scope_inst=scope_inst,
        )
        # BUG FIX: r = "" 초기화
        r = ""
        with CaptureError("strategy_kw", log_level="warning"):
            r = (
                call_gemini(client_gemini, prompt, max_tokens=600, temperature=0.7)
                if client_gemini else
                call_gpt(client_gpt, prompt, system=STRATEGY_SYSTEM,
                         max_tokens=600, model=model_gpt, temperature=0.7)
            )
        items = [
            k.strip().lstrip("•-*1234567890. ")
            for k in r.split("\n")
            if k.strip() and len(k.strip()) > 2
        ][:5]
        return items or ["분석 중 오류"]

    def _geo() -> list[str]:
        prompt = STRATEGY_GEO_TMPL.format(
            domain=domain,
            brand_name=biz.brand_name,
            site_content=site_content,
            question=rep_question,
        )
        # BUG FIX: r = "" 초기화
        r = ""
        with CaptureError("strategy_geo", log_level="warning"):
            r = (
                call_gpt(client_gpt, prompt, system=STRATEGY_SYSTEM,
                         max_tokens=1000, model=model_gpt, temperature=0.5)
                if client_gpt else
                call_gemini(client_gemini, prompt, max_tokens=1000, temperature=0.5)
            )
        items = [g.strip() for g in re.split(r'\n(?=\d+\.)', r) if g.strip()][:3]
        return items or ["분석 중 오류"]

    # ── 4개 병렬 실행 ──────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_comp = ex.submit(_competitors_fn)
        f_diag = ex.submit(_diagnosis)
        f_kw   = ex.submit(_keywords)
        f_geo  = ex.submit(_geo)
    # with 블록 종료 → executor.shutdown(wait=True) → 모든 future 완료 보장

    def _safe_result(future, fallback):
        try:
            val = future.result()  # 이미 완료 상태이므로 timeout 불필요
            return val if val is not None else fallback
        except Exception as e:
            logger.warning(f"strategy future 오류: {e}")
            return fallback

    competitors_result = _safe_result(f_comp, [])
    diagnoses          = _safe_result(f_diag, ["데이터 부족으로 분석 불가"])
    keywords           = _safe_result(f_kw,   ["분석 중 오류"])
    geo_guides         = _safe_result(f_geo,  ["분석 중 오류"])

    result = {
        "competitors": competitors_result,
        "diagnoses":   diagnoses,
        "keywords":    keywords,
        "geo_guides":  geo_guides,
    }

    if use_cache:
        cache.set(cache_key, result, namespace="strategy")

    return result
