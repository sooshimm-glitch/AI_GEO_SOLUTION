"""
구조화 로거 — 디버깅 가능한 에러 추적

수정 사항 (버그픽스):
1. logging.basicConfig 중복 호출 → force=True + Streamlit 환경 대응
2. CaptureError.__exit__ — KeyboardInterrupt/SystemExit 같은 치명적 예외는 억제하지 않음
3. get_logger 중복 핸들러 누적 방지 — 이미 핸들러가 있으면 추가하지 않음
4. safe_call fn.__module__ None 방어 — getattr 사용
"""

import logging
import traceback
import time
import sys
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps


# ── 로그 포맷 설정 ──
# BUG FIX: force=True 로 Streamlit 기존 설정 덮어쓰기
# (force는 Python 3.8+ 지원)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# BUG FIX: get_logger — 중복 핸들러 누적 방지
def get_logger(name: str) -> logging.Logger:
    """
    이미 핸들러가 설정된 logger는 그대로 반환.
    Streamlit rerun 시마다 핸들러가 중복 추가되는 문제 방지.
    """
    logger = logging.getLogger(f"aca.{name}")
    # 루트 로거에 핸들러가 있으면 전파로 충분 — 자체 핸들러 불필요
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


# ── 에러 컨텍스트 ──

@dataclass
class ErrorContext:
    operation:     str
    error:         Exception
    traceback_str: str
    timestamp:     float = field(default_factory=time.time)
    extra:         dict  = field(default_factory=dict)

    def __str__(self):
        return f"[{self.operation}] {type(self.error).__name__}: {self.error}"

    def to_dict(self) -> dict:
        return {
            "operation":  self.operation,
            "error_type": type(self.error).__name__,
            "error_msg":  str(self.error),
            "traceback":  self.traceback_str,
            "timestamp":  self.timestamp,
            **self.extra,
        }


# ── safe_call 데코레이터 ──

def safe_call(operation: str, default=None, log_level="warning"):
    """
    함수 실행 실패 시 default 반환 + 구조화 로그 출력.
    except Exception: pass 대신 이걸 사용.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # BUG FIX: getattr으로 __module__ None 방어
            module_name = getattr(fn, "__module__", None) or "unknown"
            logger = get_logger(module_name)
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                tb  = traceback.format_exc()
                ctx = ErrorContext(
                    operation=operation,
                    error=e,
                    traceback_str=tb,
                )
                log_fn = getattr(logger, log_level, logger.warning)
                log_fn(str(ctx))
                logger.debug(f"Traceback:\n{tb}")
                return default
        return wrapper
    return decorator


# ── 컨텍스트 매니저 ──

# BUG FIX: 치명적 예외 목록 — 이건 절대 억제하면 안 됨
_FATAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


class CaptureError:
    """
    with CaptureError("operation") as ctx:
        ...위험한 코드...

    if not ctx.ok:
        handle(ctx.error)

    BUG FIX:
    - KeyboardInterrupt / SystemExit / GeneratorExit 는 reraise 설정과 무관하게 항상 전파
    - reraise=False(기본)여도 치명적 예외는 억제하지 않음
    """

    def __init__(self, operation: str, reraise: bool = False, log_level: str = "warning"):
        self.operation  = operation
        self.reraise    = reraise
        self.log_level  = log_level
        self.error:     Optional[Exception]     = None
        self.error_ctx: Optional[ErrorContext]  = None
        self._logger    = get_logger("capture")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False  # 예외 없음 — 정상 종료

        # BUG FIX: 치명적 예외는 억제하지 않고 반드시 전파
        if isinstance(exc_val, _FATAL_EXCEPTIONS):
            return False  # 전파 (억제 안 함)

        # 일반 예외 처리
        self.error     = exc_val
        self.error_ctx = ErrorContext(
            operation=self.operation,
            error=exc_val,
            traceback_str=traceback.format_exc(),
        )

        log_fn = getattr(self._logger, self.log_level, self._logger.warning)
        log_fn(str(self.error_ctx))

        if self.reraise:
            return False   # 예외 전파
        return True        # 예외 억제

    @property
    def ok(self) -> bool:
        return self.error is None
