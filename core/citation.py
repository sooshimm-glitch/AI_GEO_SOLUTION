"""
문맥 인식 인용 탐지 엔진

수정 사항 (버그픽스):
1. detect_citation 패턴3 confidence 0.25 → 0.32로 올려 threshold(0.3) 이상 보장
2. _check_context_window boost 상한 0.4 → 0.35, 계산 방식 개선
3. detect_citation URL 패턴에서 find()가 -1 반환 시 잘못된 슬라이싱 방어
4. wilson_ci hits > n 방어 — p를 [0,1] 범위로 클램프
5. build_brand_variants _KO2EN/_EN2KO 중복 루프 정리
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .logger import get_logger

logger = get_logger("citation")


@dataclass
class CitationMatch:
    brand_variant:   str    # 매칭된 변형어
    pattern_type:    str    # url / entity / citation / mention
    context_snippet: str    # 매칭 전후 50자
    confidence:      float  # 0.0~1.0
    is_negative:     bool   = False
    position:        int    = -1


@dataclass
class CitationResult:
    cited:           bool
    confidence:      float
    matches:         list[CitationMatch] = field(default_factory=list)
    response_sample: str = ""

    @property
    def is_positive_citation(self) -> bool:
        return self.cited and not all(m.is_negative for m in self.matches)


# ─────────────────────────────────────────────
# 브랜드 변형 생성
# ─────────────────────────────────────────────

_EN2KO = {
    "naver": "네이버", "kakao": "카카오", "coupang": "쿠팡",
    "toss": "토스", "baemin": "배민", "krafton": "크래프톤",
    "nexon": "넥슨", "ncsoft": "엔씨소프트", "netmarble": "넷마블",
    "samsung": "삼성", "lg": "엘지", "hyundai": "현대",
    "lotte": "롯데", "sk": "에스케이", "kt": "케이티",
    "line": "라인", "nhn": "엔에이치엔", "melon": "멜론",
    "woowa": "우아한형제들", "daum": "다음", "musinsa": "무신사",
    "kurly": "마켓컬리", "yogiyo": "요기요", "gmarket": "지마켓",
    "kakaopay": "카카오페이", "kakaotalk": "카카오톡",
}
_KO2EN = {v: k for k, v in _EN2KO.items()}

_BLACKLIST = {
    "the", "inc", "co", "com", "net", "org", "ltd", "corp",
    "www", "http", "https", "서비스", "쇼핑", "앱",
}


def build_brand_variants(target_url: str, biz_info: dict) -> list[str]:
    """
    탐지 변형 목록 — 정밀도 우선 (오탐 최소화).
    최소 2자 이상, 순수 숫자·블랙리스트 제외.
    """
    try:
        p = urlparse(target_url if target_url.startswith("http") else "https://" + target_url)
        domain = p.netloc.replace("www.", "")
    except Exception:
        domain = target_url

    domain_stem = domain.split(".")[0].lower()
    brand_name  = (biz_info.get("brand_name") or "").strip()
    variants: set[str] = set()

    # 1. 도메인 기반
    variants.add(domain.lower())
    if len(domain_stem) >= 3:
        variants.add(domain_stem)

    # 2. 브랜드명 기반
    if brand_name and len(brand_name) >= 2:
        variants.add(brand_name)
        variants.add(brand_name.lower())
        no_space = brand_name.replace(" ", "")
        if len(no_space) >= 2:
            variants.add(no_space.lower())

    # BUG FIX: EN↔KO 매핑 — 중복 루프 통합
    # stem 기반 EN→KO
    if domain_stem in _EN2KO:
        variants.add(_EN2KO[domain_stem])

    # brand_name 기반 양방향 매핑
    if brand_name:
        bn_lower = brand_name.lower()
        for en, ko in _EN2KO.items():
            if en in bn_lower or en == domain_stem:
                variants.add(ko)
                variants.add(en)
        for ko, en in _KO2EN.items():
            if ko in brand_name:
                variants.add(en)
                variants.add(ko)

    # 3. 약칭 (영문 2글자 이상)
    if brand_name:
        words  = brand_name.split()
        if len(words) >= 2:
            abbrev = "".join(w[0] for w in words if w).lower()
            if len(abbrev) >= 2:
                variants.add(abbrev)

    return [
        v for v in variants
        if v and len(v) >= 2 and not v.isdigit() and v.lower() not in _BLACKLIST
    ]


# ─────────────────────────────────────────────
# 인용 패턴 정의
# ─────────────────────────────────────────────

_CITATION_PATTERNS_KO = [
    r"(?:에\s*따르면|에서\s*발표|에서\s*제공|이\s*발표한|가\s*공개한)",
    r"(?:공식\s*사이트|홈페이지|웹사이트)\s*[은는이가]",
    r"(?:서비스|플랫폼|솔루션)\s*[을를은는이가]",
    r"(?:에서\s*확인|통해\s*확인|에서\s*이용)",
    r"(?:추천|권장|사용|이용|선택)\s*(?:합니다|드립니다|할\s*수\s*있습니다)",
]

_CITATION_PATTERNS_EN = [
    r"(?:according\s+to|based\s+on|via|through|from|at)\s+\w",
    r"(?:website|platform|service|solution|tool|app)\s",
    r"(?:recommend|suggest|use|try|visit|check)",
    r"(?:official|verified|trusted|reliable)",
]

_NEGATIVE_PATTERNS_KO = [
    r"(?:하지\s*마세요|쓰지\s*마세요|피하세요|비추|불량|위험|사기|문제)",
    r"(?:단점|불편|느림|비쌈|최악)",
    r"(?:보다\s*나쁜|보다\s*못한|대신\s*사용)",
]

_NEGATIVE_PATTERNS_EN = [
    r"(?:avoid|don't\s+use|not\s+recommended|scam|fraud|poor|worst|terrible)",
    r"(?:instead\s+of|rather\s+than|unlike)",
]

_ALL_CITATION = _CITATION_PATTERNS_KO + _CITATION_PATTERNS_EN
_ALL_NEGATIVE = _NEGATIVE_PATTERNS_KO + _NEGATIVE_PATTERNS_EN


def _check_context_window(
    text: str, pos: int, window: int = 80
) -> tuple[str, bool, float]:
    """
    브랜드명 위치(pos) 주변 window 문자 내 문맥 분석.
    반환: (snippet, is_negative, confidence_boost)
    """
    # BUG FIX: pos=-1(find 실패) 방어
    if pos < 0:
        return "", False, 0.0

    start   = max(0, pos - window)
    end     = min(len(text), pos + window)
    snippet = text[start:end]

    # 부정 패턴 체크
    for pat in _ALL_NEGATIVE:
        if re.search(pat, snippet, re.IGNORECASE):
            return snippet, True, 0.0

    # BUG FIX: 인용 패턴 boost — 매칭 수에 비례, 상한 0.35
    boost       = 0.0
    match_count = 0
    for pat in _ALL_CITATION:
        if re.search(pat, snippet, re.IGNORECASE):
            match_count += 1
    boost = min(match_count * 0.15, 0.35)

    return snippet, False, boost


def detect_citation(
    response: str,
    brand_variants: list[str],
    threshold: float = 0.3,
) -> CitationResult:
    """
    문맥 인식 인용 탐지.

    탐지 우선순위:
    1. URL 패턴 매칭           (confidence: 0.9)
    2. 단어 경계 매칭 + 문맥   (0.55~0.9)
    3. 단순 포함 + 문맥 검증   (0.32~0.67)  ← BUG FIX: 0.25→0.32

    threshold 미만이면 cited=False.
    """
    if not response or not brand_variants:
        return CitationResult(cited=False, confidence=0.0)

    response_lower  = response.lower()
    best_confidence = 0.0
    all_matches: list[CitationMatch] = []

    for variant in brand_variants:
        if not variant or len(variant) < 2:
            continue

        v_lower = variant.lower()

        # ── 패턴 1: URL 매칭 (최고 신뢰도) ──
        url_pat   = re.escape(v_lower)
        url_match = re.search(
            rf'https?://[^\s]*{url_pat}|{url_pat}\.[a-z]{{2,6}}',
            response_lower
        )
        if url_match:
            pos = url_match.start()
            # BUG FIX: regex 매칭 위치 사용 (find() 대신)
            snippet, is_neg, boost = _check_context_window(response_lower, pos)
            conf = 0.0 if is_neg else min(0.9 + boost, 1.0)
            all_matches.append(CitationMatch(
                brand_variant=variant,
                pattern_type="url",
                context_snippet=snippet,
                confidence=conf,
                is_negative=is_neg,
                position=pos,
            ))
            best_confidence = max(best_confidence, conf)
            continue

        # ── 패턴 2: 단어 경계 매칭 ──
        is_korean   = bool(re.search(r'[가-힣]', v_lower))
        if is_korean:
            boundary_pat = rf'(?<![가-힣a-z0-9]){re.escape(v_lower)}(?![가-힣a-z0-9])'
        else:
            boundary_pat = rf'\b{re.escape(v_lower)}\b'

        for m_obj in re.finditer(boundary_pat, response_lower, re.IGNORECASE):
            pos     = m_obj.start()
            snippet, is_neg, boost = _check_context_window(response_lower, pos)
            conf    = 0.0 if is_neg else (0.55 + boost)
            all_matches.append(CitationMatch(
                brand_variant=variant,
                pattern_type="entity",
                context_snippet=snippet,
                confidence=conf,
                is_negative=is_neg,
                position=pos,
            ))
            best_confidence = max(best_confidence, conf)
            break  # 첫 매칭만

        if best_confidence >= 0.6:
            continue  # 충분히 확신, 다음 변형 체크 불필요

        # ── 패턴 3: 단순 포함 (낮은 신뢰도, 문맥 검증 필수) ──
        idx = response_lower.find(v_lower)
        if idx >= 0:
            snippet, is_neg, boost = _check_context_window(response_lower, idx)
            # BUG FIX: 기본 confidence 0.25 → 0.32 (threshold 0.3 이상 보장)
            conf = 0.0 if is_neg else (0.32 + boost)
            all_matches.append(CitationMatch(
                brand_variant=variant,
                pattern_type="mention",
                context_snippet=snippet,
                confidence=conf,
                is_negative=is_neg,
                position=idx,
            ))
            best_confidence = max(best_confidence, conf)

    cited      = best_confidence >= threshold
    best_match = max(all_matches, key=lambda x: x.confidence) if all_matches else None

    return CitationResult(
        cited=cited,
        confidence=best_confidence,
        matches=all_matches,
        response_sample=(
            best_match.context_snippet if best_match and cited else ""
        ),
    )


# ─────────────────────────────────────────────
# 통계 유틸
# ─────────────────────────────────────────────

def wilson_ci(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson Score Interval — 소표본에서도 안정적."""
    if n == 0:
        return 0.0, 100.0

    # BUG FIX: 조기종료 추정으로 hits > n 가능 → p를 [0,1] 클램프
    hits = max(0, min(hits, n))
    p    = hits / n

    z      = 1.96 if confidence == 0.95 else 2.576
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom

    # p*(1-p) 가 음수가 되지 않도록 보장
    variance = max(0.0, p * (1 - p) / n + z**2 / (4 * n**2))
    margin   = (z * math.sqrt(variance)) / denom

    return (
        round(max(0.0,   (center - margin) * 100), 1),
        round(min(100.0, (center + margin) * 100), 1),
    )
