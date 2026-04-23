"""
캐싱 레이어 — TTL 기반 인메모리 캐시

비용 폭탄 방지: 동일 URL/질문 결과를 TTL(기본 1시간) 동안 재사용

수정 사항 (버그픽스):
1. from_serializable — 만료된 항목 복원 방지 (created_at + ttl 체크)
2. to_serializable — dataclass/객체 직렬화 불가 시 TypeError 방어
3. stats() — entries 카운트에서 만료 항목 제외
4. get() 호출 시 자동 만료 정리 (일정 확률로 clear_expired 실행)
5. make_key namespace 중복 전달 문제 — 사용 방식 주석으로 명확히 문서화
"""

import hashlib
import time
import json
import threading
from dataclasses import dataclass, asdict, fields
from typing import Any, Optional


@dataclass
class CacheEntry:
    value:      Any
    created_at: float
    ttl:        float   # seconds

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


def _safe_serialize(value: Any) -> Any:
    """
    직렬화 불가능한 객체를 안전하게 변환.
    dataclass → dict, 나머지는 그대로.
    """
    if value is None:
        return None
    # dataclass 인스턴스
    try:
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
    except Exception:
        pass
    # list/tuple 내부 재귀
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    # dict 내부 재귀
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}
    # JSON 직렬화 가능 여부 확인
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        # 직렬화 불가 → 문자열로 변환 (최후 수단)
        return str(value)


class TTLCache:
    """
    Thread-safe TTL 캐시.

    네임스페이스별 TTL:
    - sim:         3600s  (시뮬레이션 결과)
    - biz:         86400s (비즈니스 분석, 하루)
    - competitors: 86400s (경쟁사 목록, 하루)
    - crawl:       3600s  (크롤링 결과)
    - strategy:    7200s  (전략 분석)

    사용 패턴:
        key = cache.make_key("sim", url, question, ...)  # namespace를 키에 포함
        cache.set(key, value, namespace="sim")            # TTL 결정용 namespace 별도 전달
    """

    DEFAULT_TTL = {
        "sim":         3600,
        "biz":         86400,
        "competitors": 86400,
        "crawl":       3600,
        "strategy":    7200,
    }

    # get() 호출 N번마다 자동 만료 정리 실행
    _AUTO_CLEAN_INTERVAL = 50

    def __init__(self):
        self._store:    dict[str, CacheEntry] = {}
        self._lock      = threading.RLock()
        self._hits      = 0
        self._misses    = 0
        self._get_count = 0  # 자동 정리용 카운터

    # ── 키 생성 ──

    @staticmethod
    def make_key(namespace: str, *parts) -> str:
        """
        네임스페이스 + 가변 인자로 고유 키 생성.
        namespace는 키 해시에 포함되므로 set()의 namespace와 일치시켜야 함.
        """
        raw = namespace + "|" + "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ── CRUD ──

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._get_count += 1

            # 자동 만료 정리 (일정 주기)
            if self._get_count % self._AUTO_CLEAN_INTERVAL == 0:
                self._do_clear_expired()

            entry = self._store.get(key)
            if entry is None or entry.is_expired:
                if entry:
                    del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, namespace: str = "sim") -> None:
        ttl = self.DEFAULT_TTL.get(namespace, 3600)
        with self._lock:
            self._store[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl,
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def _do_clear_expired(self) -> int:
        """락 내부에서 호출 (이미 락 보유 상태)"""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    def clear_expired(self) -> int:
        """외부에서 호출 가능한 만료 정리"""
        with self._lock:
            return self._do_clear_expired()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            # BUG FIX: 만료 항목 제외한 실제 유효 entries 카운트
            valid_entries = sum(1 for v in self._store.values() if not v.is_expired)
            return {
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": round(self._hits / total * 100, 1) if total else 0.0,
                "entries":  valid_entries,
            }

    def to_serializable(self) -> dict:
        """
        Streamlit session_state 저장용 직렬화.
        BUG FIX: dataclass/객체 value를 안전하게 직렬화
        """
        with self._lock:
            result = {}
            for k, v in self._store.items():
                if v.is_expired:
                    continue
                try:
                    serialized_value = _safe_serialize(v.value)
                    # JSON 직렬화 가능 여부 최종 검증
                    json.dumps(serialized_value)
                    result[k] = {
                        "value":      serialized_value,
                        "created_at": v.created_at,
                        "ttl":        v.ttl,
                    }
                except (TypeError, ValueError):
                    # 직렬화 끝까지 실패하면 해당 항목 건너뜀
                    continue
            return result

    @classmethod
    def from_serializable(cls, data: dict) -> "TTLCache":
        """
        Streamlit session_state 복원.
        BUG FIX: 복원 시 이미 만료된 항목 제외
        """
        cache = cls()
        now   = time.time()
        for k, v in data.items():
            created_at = v.get("created_at", 0)
            ttl        = v.get("ttl", 3600)
            # 만료된 항목은 복원하지 않음
            if now - created_at > ttl:
                continue
            cache._store[k] = CacheEntry(
                value=v.get("value"),
                created_at=created_at,
                ttl=ttl,
            )
        return cache


# 전역 싱글톤
_global_cache = TTLCache()


def get_cache() -> TTLCache:
    return _global_cache
