"""
공유 데이터 스키마 — 엄격한 타입 보장 + 안전한 역직렬화

수정 사항 (버그픽스):
1. BusinessInfo.__post_init__ crawl_tier 타입 변환 — float/음수/None 모두 안전 처리
2. Competitor.__post_init__ rank 문자열 혼합형 방어 — int() 변환 실패 시 기본값
3. Competitor.to_dict() — __dict__ 대신 dataclasses.asdict() 사용
4. BusinessInfo.from_dict — None 값 필드 방어 처리
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


# ── 업종 카테고리 정의 ──────────────────────────────────────────────

CATEGORY_TO_DETAIL: dict[str, list[str]] = {
    "광고/마케팅": [
        "퍼포먼스 마케팅 대행사", "종합 광고대행사", "콘텐츠 마케팅 에이전시",
        "검색광고(SA) 전문 대행사", "소셜미디어 마케팅 대행사", "PR·홍보 대행사",
        "인플루언서 마케팅 플랫폼", "프로그래매틱 광고 플랫폼",
    ],
    "이커머스": [
        "종합 온라인 쇼핑몰", "버티컬 패션 이커머스", "신선식품 새벽배송",
        "오픈마켓 플랫폼", "라이브커머스 플랫폼", "B2B 도매 마켓플레이스",
    ],
    "SaaS": [
        "B2B HR SaaS", "CRM/영업 자동화 SaaS", "회계/ERP SaaS",
        "마케팅 자동화 SaaS", "프로젝트 관리 협업툴", "데이터 분석 BI 플랫폼",
        "보안/제로트러스트 SaaS", "클라우드 인프라",
    ],
    "금융": [
        "인터넷전문은행", "핀테크 간편결제", "P2P 대출 플랫폼",
        "로보어드바이저 투자", "보험 비교 플랫폼", "가상자산 거래소",
    ],
    "교육": [
        "온라인 코딩 교육", "자격증/공무원 인강", "영어 회화 플랫폼",
        "대학입시 전문 학원", "성인 직무 역량 교육", "어린이 STEM 교육",
    ],
    "의료": [
        "비대면 진료 플랫폼", "병원 예약·리뷰 플랫폼", "의약품 유통",
        "헬스케어 웨어러블", "정신건강 상담 플랫폼",
    ],
    "게임": [
        "모바일 RPG", "PC 온라인 FPS", "캐주얼 하이퍼게임",
        "게임 배급·퍼블리싱", "e스포츠 플랫폼",
    ],
    "미디어": [
        "OTT 스트리밍", "뉴스 미디어", "팟캐스트 플랫폼",
        "웹툰/웹소설 플랫폼", "음악 스트리밍",
    ],
    "부동산": [
        "부동산 직거래 플랫폼", "프롭테크 중개 서비스", "상업용 부동산 리서치",
    ],
    "제조": [
        "소비자가전 제조", "자동차 부품 제조", "반도체 설계/파운드리",
        "식품 제조", "뷰티/화장품 OEM",
    ],
    "물류": [
        "풀필먼트 3PL", "라스트마일 배송", "B2B 화물 중개 플랫폼",
    ],
    "기타": ["기타 서비스"],
}

INDUSTRY_CATEGORIES: list[str] = list(CATEGORY_TO_DETAIL.keys())

VALID_CONFIDENCES      = {"high", "medium", "low"}
VALID_MARKET_POSITIONS = {"업계 1위", "신흥 강자", "틈새 전문"}

_BAD_INDUSTRIES: frozenset[str] = frozenset({
    "디지털 서비스", "it 서비스", "온라인 서비스",
    "인터넷 서비스", "웹 서비스", "소프트웨어",
    "technology", "tech company", "internet company",
})


def _safe_int(value: Any, default: int = 0) -> int:
    """
    BUG FIX: int 변환 헬퍼 — float, 문자열 혼합형, None 모두 안전 처리.
    예) 1.0 → 1, "2위" → default, None → default, -1 → -1(그대로)
    """
    if value is None:
        return default
    try:
        # float → int (1.0 → 1)
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        # "2위" 같은 혼합형 — 앞의 숫자만 추출
        m = re.match(r'[-+]?\d+', str(value).strip())
        if m:
            return int(m.group())
        return default


# ── BusinessInfo ────────────────────────────────────────────────────

@dataclass
class BusinessInfo:
    brand_name:        str
    industry:          str
    industry_category: str
    core_product:      str
    target_audience:   str
    key_services:      list[str] = field(default_factory=list)
    confidence:        str       = "medium"
    crawl_tier:        int       = 0

    def __post_init__(self) -> None:
        # 문자열 필드 안전 변환
        self.brand_name        = str(self.brand_name        or "").strip() or "Unknown"
        self.industry          = str(self.industry          or "").strip() or "기타 서비스"
        self.industry_category = str(self.industry_category or "").strip()
        self.core_product      = str(self.core_product      or "").strip() or self.industry
        self.target_audience   = str(self.target_audience   or "").strip() or "잠재 고객"
        self.confidence        = self.confidence if self.confidence in VALID_CONFIDENCES else "medium"

        # BUG FIX: crawl_tier — float/문자열/None 모두 안전하게 처리
        self.crawl_tier = max(0, _safe_int(self.crawl_tier, default=0))

        # key_services: list[str] 보장
        if not isinstance(self.key_services, list):
            self.key_services = []
        self.key_services = [str(s) for s in self.key_services if s][:5]

        # industry_category 유효성
        if self.industry_category not in INDUSTRY_CATEGORIES:
            self.industry_category = "기타"

    def is_vague_industry(self) -> bool:
        return any(bad in self.industry.lower() for bad in _BAD_INDUSTRIES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brand_name":        self.brand_name,
            "industry":          self.industry,
            "industry_category": self.industry_category,
            "core_product":      self.core_product,
            "target_audience":   self.target_audience,
            "key_services":      self.key_services,
            "confidence":        self.confidence,
            "crawl_tier":        self.crawl_tier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BusinessInfo":
        """
        알 수 없는 키 무시, 누락 키 기본값으로 안전 복원.
        BUG FIX: None 값 필드도 __post_init__ 에서 처리되도록 그대로 전달
        """
        known = set(cls.__dataclass_fields__)
        safe  = {k: v for k, v in d.items() if k in known}
        return cls(**safe)

    @classmethod
    def fallback(cls, domain_stem: str, crawl_tier: int = 0) -> "BusinessInfo":
        brand = domain_stem.replace("-", " ").replace("_", " ").title()
        return cls(
            brand_name=brand,
            industry=f"{brand} 관련 서비스",
            industry_category="기타",
            core_product=f"{brand} 서비스",
            target_audience="잠재 고객",
            confidence="low",
            crawl_tier=crawl_tier,
        )


# ── Competitor ──────────────────────────────────────────────────────

@dataclass
class Competitor:
    rank:            int
    brand_name:      str
    domain:          str
    reason:          str
    market_position: str
    verified:        bool = False
    evidence:        str  = ""

    def __post_init__(self) -> None:
        # BUG FIX: rank — 문자열 혼합형("1위" 등) 안전 변환
        self.rank           = max(1, _safe_int(self.rank, default=1))
        self.brand_name     = str(self.brand_name     or "").strip()
        self.domain         = str(self.domain         or "").strip().lower()
        self.reason         = str(self.reason         or "")[:60]
        self.market_position = (
            self.market_position
            if self.market_position in VALID_MARKET_POSITIONS
            else "틈새 전문"
        )
        self.evidence = str(self.evidence or "")[:200]

    @staticmethod
    def is_fake_domain(domain: str) -> bool:
        """가짜/더미 도메인 빠른 필터."""
        if not domain or len(domain) < 4:
            return True
        if re.match(r'^(competitor|example|test|dummy|fake|c)\d*\.', domain):
            return True
        if not re.search(r'\.[a-z]{2,6}$', domain):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        # BUG FIX: __dict__ 대신 dataclasses.asdict() 사용 — 더 안전한 직렬화
        try:
            return asdict(self)
        except Exception:
            # asdict 실패 시 (중첩 비직렬화 객체 등) __dict__ 폴백
            return self.__dict__.copy()
