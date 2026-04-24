"""
문맥 인식 인용 탐지 엔진 v2.2

버그 수정:
  [BUG 3 FIX] build_brand_variants — 하이픈 도메인 미분리
              avahair-avenue.com → avahair, avenue 개별 추출 안 됨
              → 하이픈 분리 + 한글 브랜드 공백 변형 추가
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
    brand_variant:    str    # 매칭된 변형어
    pattern_type:     str    # url / entity / citation / mention
    context_snippet:  str    # 매칭 전후 80자
    confidence:       float  # 0.0~1.0
    is_negative:      bool = False
    position:         int  = -1


@dataclass
class CitationResult:
    cited:      bool
    confidence: float
    matches:    list = field(default_factory=list)
    response_sample: str = ""

    @property
    def is_positive_citation(self) -> bool:
        return self.cited and not all(m.is_negative for m in self.matches)


# ─────────────────────────────────────────────
# EN↔KO 매핑
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

# 너무 일반적인 단어는 하이픈 파트로 추가하지 않음
_GENERIC_WORDS = {
    "hair", "shop", "store", "cafe", "bar", "salon", "house",
    "home", "art", "studio", "lab", "hub", "zone", "space",
    "pro", "plus", "max", "go", "one", "two", "app",
}


def build_brand_variants(target_url: str, biz_info: dict) -> list[str]:
    """
    탐지 변형 목록 — 정밀도 + 재현율 균형.

    [BUG 3 FIX] 하이픈 도메인 처리 추가:
      avahair-avenue.com
        → avahair-avenue (원본 스템)
        → avahair        (첫 번째 파트, 브랜드 핵심어)
        → avahairavenue  (하이픈 제거 합성)
        avenue는 _GENERIC_WORDS에 없으므로 추가

    [추가] 한글 브랜드명 공백 변형:
      에이바헤어 → 에이바 헤어 (공백 중간 삽입)
    """
    try:
        p = urlparse(target_url if target_url.startswith("http") else "https://" + target_url)
        domain = p.netloc.replace("www.", "")
    except Exception:
        domain = target_url

    domain_stem = domain.split(".")[0].lower()   # e.g. "avahair-avenue"
    brand_name  = (biz_info.get("brand_name") or "").strip()

    variants: set[str] = set()

    # ── 1. 도메인 기반 ──
    variants.add(domain.lower())
    if len(domain_stem) >= 3:
        variants.add(domain_stem)

    # ── 2. [FIX] 하이픈 도메인 분리 ──
    if "-" in domain_stem:
        parts = domain_stem.split("-")

        # 2-1. 하이픈 제거 합성: avahair-avenue → avahairavenue
        joined = "".join(parts)
        if len(joined) >= 3:
            variants.add(joined)

        # 2-2. 첫 번째 파트 (브랜드 핵심어): avahair
        if len(parts[0]) >= 3:
            variants.add(parts[0])

        # 2-3. 나머지 파트 — 일반어 블랙리스트 제외 후 추가
        for part in parts[1:]:
            if len(part) >= 3 and part not in _BLACKLIST and part not in _GENERIC_WORDS:
                variants.add(part)

    # ── 3. 브랜드명 기반 ──
    if brand_name and len(brand_name) >= 2:
        variants.add(brand_name)
        variants.add(brand_name.lower())

        # 공백 제거 버전
        no_space = brand_name.replace(" ", "")
        if len(no_space) >= 2:
            variants.add(no_space.lower())

        # [추가] 한글 브랜드명 공백 삽입 변형
        # "에이바헤어" → "에이바 헤어", "에이 바헤어"
        if re.search(r'[가-힣]', brand_name) and len(brand_name) >= 4:
            mid = len(brand_name) // 2
            for split_at in [mid - 1, mid, mid + 1]:
                if 2 <= split_at <= len(brand_name) - 2:
                    spaced = brand_name[:split_at] + " " + brand_name[split_at:]
                    if len(spaced) >= 3:
                        variants.add(spaced)

    # ── 4. EN↔KO 매핑 ──
    # 스템 및 하이픈 파트 기반
    check_parts = [domain_stem]
    if "-" in domain_stem:
        check_parts += domain_stem.split("-")

    for part in check_parts:
        if part in _EN2KO:
            variants.add(_EN2KO[part])

    # 브랜드명 기반 양방향
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

    # ── 5. 약칭 (영문 2글자 이상) ──
    if brand_name:
        words = brand_name.split()
        if len(words) >= 2:
            abbrev = "".join(w[0] for w in words if w).lower()
            if len(abbrev) >= 2:
                variants.add(abbrev)

    # ── 필터: 최소 2자, 순수 숫자 제외, 블랙리스트 제외 ──
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
    r"(?:서비스|플랫폼|솔루션|매장|가게|미용실|헤어샵)\s*[을를은는이가]",
    r"(?:에서\s*확인|통해\s*확인|에서\s*이용|방문)",
    r"(?:추천|권장|사용|이용|선택|예약)\s*(?:합니다|드립니다|할\s*수\s*있습니다|해보세요)",
    r"(?:위치|주소|전화|영업)",
]

_CITATION_PATTERNS_EN = [
    r"(?:according\s+to|based\s+on|via|through|from|at)\s+\w",
    r"(?:website|platform|service|solution|tool|app|salon|shop)\s",
    r"(?:recommend|suggest|use|try|visit|check|book)",
    r"(?:official|verified|trusted|reliable|nearby|located)",
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


def _check_context_window(text: str, pos: int, window: int = 80) -> tuple:
    if pos < 0:
        return "", False, 0.0

    start   = max(0, pos - window)
    end     = min(len(text), pos + window)
    snippet = text[start:end]

    for pat in _ALL_NEGATIVE:
        if re.search(pat, snippet, re.IGNORECASE):
            return snippet, True, 0.0

    match_count = 0
    for pat in _ALL_CITATION:
        if re.search(pat, snippet, re.IGNORECASE):
            match_count += 1
    boost = min(match_count * 0.15, 0.35)

    return snippet, False, boost


def detect_citation(
    response: str,
    brand_variants: list,
    threshold: float = 0.3,
) -> CitationResult:
    if not response or not brand_variants:
        return CitationResult(cited=False, confidence=0.0)

    response_lower   = response.lower()
    best_confidence  = 0.0
    all_matches: list = []

    for variant in brand_variants:
        if not variant or len(variant) < 2:
            continue
        v_lower = variant.lower()

        # ── 패턴 1: URL 매칭 (최고 신뢰도) ──
        url_match = re.search(
            rf'https?://[^\s]*{re.escape(v_lower)}|{re.escape(v_lower)}\.[a-z]{{2,6}}',
            response_lower,
        )
        if url_match:
            pos = url_match.start()
            snippet, is_neg, boost = _check_context_window(response_lower, pos)
            conf = 0.0 if is_neg else min(0.9 + boost, 1.0)
            all_matches.append(CitationMatch(
                brand_variant=variant, pattern_type="url",
                context_snippet=snippet, confidence=conf,
                is_negative=is_neg, position=pos,
            ))
            best_confidence = max(best_confidence, conf)
            continue

        # ── 패턴 2: 단어 경계 매칭 ──
        is_korean = bool(re.search(r'[가-힣]', v_lower))
        if is_korean:
            boundary_pat = rf'(?<![가-힣a-z0-9]){re.escape(v_lower)}(?![가-힣a-z0-9])'
        else:
            boundary_pat = rf'\b{re.escape(v_lower)}\b'

        for m_obj in re.finditer(boundary_pat, response_lower, re.IGNORECASE):
            pos = m_obj.start()
            snippet, is_neg, boost = _check_context_window(response_lower, pos)
            conf = 0.0 if is_neg else (0.55 + boost)
            all_matches.append(CitationMatch(
                brand_variant=variant, pattern_type="entity",
                context_snippet=snippet, confidence=conf,
                is_negative=is_neg, position=pos,
            ))
            best_confidence = max(best_confidence, conf)
            break

        if best_confidence >= 0.6:
            continue

        # ── 패턴 3: 단순 포함 (기본 confidence 0.32 — threshold 0.3 이상) ──
        idx = response_lower.find(v_lower)
        if idx >= 0:
            snippet, is_neg, boost = _check_context_window(response_lower, idx)
            conf = 0.0 if is_neg else (0.32 + boost)
            all_matches.append(CitationMatch(
                brand_variant=variant, pattern_type="mention",
                context_snippet=snippet, confidence=conf,
                is_negative=is_neg, position=idx,
            ))
            best_confidence = max(best_confidence, conf)

    cited      = best_confidence >= threshold
    best_match = max(all_matches, key=lambda x: x.confidence) if all_matches else None

    return CitationResult(
        cited=cited,
        confidence=best_confidence,
        matches=all_matches,
        response_sample=(best_match.context_snippet if best_match and cited else ""),
    )


# ─────────────────────────────────────────────
# 통계 유틸
# ─────────────────────────────────────────────

def wilson_ci(hits: int, n: int, confidence: float = 0.95) -> tuple:
    if n == 0:
        return 0.0, 100.0
    hits = max(0, min(hits, n))
    p    = hits / n
    z    = 1.96 if confidence == 0.95 else 2.576

    denom    = 1 + z**2 / n
    center   = (p + z**2 / (2 * n)) / denom
    variance = max(0.0, p * (1 - p) / n + z**2 / (4 * n**2))
    margin   = (z * math.sqrt(variance)) / denom

    return (
        round(max(0.0,   (center - margin) * 100), 1),
        round(min(100.0, (center + margin) * 100), 1),
    )
