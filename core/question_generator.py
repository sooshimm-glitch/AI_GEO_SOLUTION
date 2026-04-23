"""
question_generator.py — 타겟 질문 생성 전담 모듈

수동 분석형(Tab2): 브랜드명 미포함, 카테고리 전체 대상 질문 생성.
자동 분석형(Tab1)은 pipeline.py의 generate_questions_from_state() 사용.

수정 사항 (버그픽스):
1. prompts.py import 제거 — 파일 없으면 ImportError로 앱 시작 불가
   → 프롬프트를 이 파일 내부에 직접 정의
2. _parse_questions bold 제거 regex 이중 이스케이프 오류 수정
3. generate_target_questions 에서 ctx.ok=False 시 raise 제거
   → 에러 로그만 남기고 fallback 질문으로 대체
4. cache 히트 체크 `if cached:` → `if cached is not None:` 수정
   (빈 리스트 캐시도 정상 히트로 처리)
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from .logger import get_logger, CaptureError
from .cache import get_cache
from .ai_client import call_gpt, call_gemini
from .schemas import BusinessInfo

logger = get_logger("question_generator")


# ─────────────────────────────────────────────
# 프롬프트 상수 (prompts.py 의존성 제거)
# ─────────────────────────────────────────────

QUESTION_SYSTEM = (
    "당신은 GEO(Generative Engine Optimization) 전문가이자 디지털 마케팅 전략가입니다. "
    "완성된 질문만 출력합니다."
)

# 수동 분석형 질문 생성 템플릿
QUESTION_GEN_MANUAL_TMPL = """당신은 {industry} 분야 10년 경력 마케팅 전략가이자 GEO 전문가입니다.

[분석 대상 서비스 정보 — 내부 참고용, 질문에 직접 노출 금지]
- 업종: {industry} ({industry_category})
- 핵심 서비스: {services_str}
- 주요 타겟: {target_audience}

[질문 방향]
{category_hint}

[핵심 규칙 — 반드시 준수]
1. 실제 사용자가 네이버, 구글, ChatGPT 검색창에 입력하는 방식의 자연스러운 질문
2. 브랜드명, 회사명, 도메인 주소를 질문에 절대 포함하지 말 것
3. "{industry}" 카테고리 전체를 대상으로 하는 질문 (특정 브랜드 X)
4. 구매 결정 5단계(인지, 비교, 신뢰, 가격, 전환) 각각 다룰 것
5. {industry} 업계 전문 용어와 지표 적극 활용
6. 기초 탐색형 질문 금지 ("무엇인가요?", "소개해주세요" 등)

번호 없이 질문 5개만 출력. 한 줄에 하나. 물음표(?)로 종결."""


# biz_analysis.py 기존 import 경로 호환용 re-export
_QUESTION_SYSTEM = QUESTION_SYSTEM

_CATEGORY_HINTS: dict[str, str] = {
    "광고/마케팅": (
        "광고주(중소기업 대표, 마케터)가 대행사를 선택할 때 묻는 질문 — "
        "ROAS, CPA, 매체비, 대행수수료, 업종 레퍼런스 위주"
    ),
    "이커머스": "구매자의 배송·가격·신뢰도, 판매자의 입점·수수료 관련 질문",
    "SaaS":    "도입 전 데모·연동·보안·가격 플랜, 기존 솔루션 전환 비용 관련 질문",
    "금융":    "금리·한도·수수료·안전성, 타 금융사 대비 혜택 관련 질문",
    "교육":    "커리큘럼·강사·합격률·환불정책, 취업 연계 관련 질문",
    "의료":    "진료 과목·비용·예약, 전문성 관련 질문",
    "게임":    "게임성·과금정책·PC/모바일 지원, 경쟁 타이틀 대비 질문",
}

_QUESTION_VERSION = "v4"


def generate_target_questions(
    client_gpt,
    client_gemini,
    url:       str,
    engine:    str,
    model_gpt: str,
    biz_info:  Optional[BusinessInfo] = None,
    use_cache: bool = True,
) -> list[str]:
    """
    비즈니스 정보 기반 고품질 타겟 질문 5개 생성.
    브랜드명을 포함하지 않는 카테고리 전체 대상 질문 (수동 분석형 Tab2용).
    """
    try:
        p      = urlparse(url if url.startswith("http") else "https://" + url)
        domain = p.netloc.replace("www.", "")
    except Exception:
        domain = url

    if biz_info is None:
        biz_info = BusinessInfo(
            brand_name=domain.split(".")[0].upper(),
            industry="서비스",
            industry_category="기타",
            core_product="서비스",
            target_audience="잠재 고객",
        )

    cache     = get_cache()
    cache_key = cache.make_key(
        "questions", _QUESTION_VERSION, url, biz_info.industry, engine, model_gpt
    )

    # BUG FIX: `if cached:` → `if cached is not None:` (빈 리스트 캐시도 히트 처리)
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache HIT: questions({domain})")
            return cached

    category_hint = _CATEGORY_HINTS.get(
        biz_info.industry_category,
        f"{biz_info.industry} 분야에서 {biz_info.target_audience}가 실제로 고민하는 핵심 질문",
    )
    services_str = (
        ", ".join(biz_info.key_services) if biz_info.key_services else biz_info.core_product
    )

    # BUG FIX: prompts.py 의존성 제거 — 내부 템플릿 직접 사용
    prompt = QUESTION_GEN_MANUAL_TMPL.format(
        industry=biz_info.industry,
        industry_category=biz_info.industry_category,
        services_str=services_str,
        target_audience=biz_info.target_audience,
        category_hint=category_hint,
    )

    result_str = ""
    with CaptureError("question_gen", log_level="warning") as ctx:
        if engine == "GPT" and client_gpt:
            result_str = call_gpt(
                client_gpt, prompt, system=QUESTION_SYSTEM,
                max_tokens=600, model=model_gpt, temperature=0.85,
            )
        elif engine == "Gemini" and client_gemini:
            result_str = call_gemini(client_gemini, prompt, max_tokens=600, temperature=0.85)
        elif client_gpt:
            # 요청 엔진 사용 불가 시 GPT로 폴백
            result_str = call_gpt(
                client_gpt, prompt, system=QUESTION_SYSTEM,
                max_tokens=600, model=model_gpt, temperature=0.85,
            )
        elif client_gemini:
            # GPT도 없으면 Gemini로 폴백
            result_str = call_gemini(client_gemini, prompt, max_tokens=600, temperature=0.85)

    # BUG FIX: ctx.ok=False 시 raise 제거 → 에러 로그 후 fallback으로 대체
    if not ctx.ok:
        logger.warning(f"질문 생성 실패 ({domain}): {ctx.error} — 폴백 사용")

    questions = _parse_questions(result_str) if result_str else []
    if len(questions) < 3:
        questions = _fallback_questions(biz_info)

    result = questions[:5]
    if use_cache and result:
        cache.set(cache_key, result, namespace="biz")

    return result


def _parse_questions(raw: str) -> list[str]:
    out: list[str] = []
    for ln in raw.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        # BUG FIX: bold 제거 — 올바른 regex (이중 이스케이프 없이)
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', ln)   # **bold** → 내용만
        clean = re.sub(r'^\d+[.)]\s*', '', clean)      # 앞머리 번호 제거
        clean = re.sub(r'^[-•*]\s*', '', clean)         # 앞머리 bullet 제거
        clean = re.sub(r'^\[.*?\]\s*', '', clean).strip()  # 앞머리 [태그] 제거
        if len(clean) > 10:
            if not clean.endswith("?"):
                clean += "?"
            out.append(clean)
    return out


def _fallback_questions(biz: BusinessInfo) -> list[str]:
    is_ad = "광고" in biz.industry or "마케팅" in biz.industry
    if is_ad:
        return [
            f"{biz.industry} 대행사 선택할 때 업종별 ROAS 기준으로 비교하는 방법은?",
            "광고대행사 계약 전 반드시 확인해야 할 조건과 주의사항은?",
            f"{biz.industry} 집행 매체별 효율 차이와 예산 배분 기준은?",
            "직접 광고 운영 vs 대행사 위탁 — 비용 구조와 성과 차이 비교는?",
            f"{biz.industry} 실제 집행 성과 사례와 평균 CPA 수준은 어느 정도인가요?",
        ]
    return [
        f"{biz.industry} 서비스 선택할 때 경쟁사 대비 꼭 확인해야 할 차이점은?",
        f"{biz.target_audience}가 {biz.industry} 도입 후 실제로 얻은 성과 사례는?",
        f"{biz.industry} 계약 및 이용 조건, 비용 구조가 업계 평균과 비교해 어떤가요?",
        f"{biz.industry} 실제 사용자 평가에서 자주 언급되는 불만 사항은?",
        f"{biz.industry} 서비스를 비교할 때 가장 중요한 선택 기준은 무엇인가요?",
    ]
