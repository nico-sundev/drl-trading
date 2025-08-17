"""Generic timing utilities (decorators & context managers).

Design goals:
 - Zero third‑party dependency (just time + logging)
 - Structured emission hook (user can provide a callable to collect metrics)
 - Minimal overhead when disabled
 - Works for sync functions and code blocks
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast, Dict
import time
import logging

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timing(
    name: Optional[str] = None,
    *,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    logger_: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    enabled: bool = True,
) -> Callable[[F], F]:
    """Decorator to measure execution time of a function.

    Args:
        name: Optional logical name; defaults to function __qualname__
        emit: Optional callback receiving a metric dict
        logger_: Optional logger (defaults to module logger)
        level: Log level for emission when using logger
        enabled: Fast toggle; when False it becomes a no-op
    """

    def decorator(fn: F) -> F:
        if not enabled:
            return fn

        metric_name = name or fn.__qualname__
        log = logger_ or logger

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                duration_ms = round((time.time() - start) * 1000, 3)
                payload = {"metric": "timing", "name": metric_name, "duration_ms": duration_ms}
                if emit:
                    try:
                        emit(payload)
                    except Exception:  # pragma: no cover - defensive
                        log.debug("timing emit callback failed", exc_info=True)
                else:
                    if log.isEnabledFor(level):
                        log.log(level, "TIMING %s %sms", metric_name, duration_ms)

        return cast(F, wrapper)

    return decorator


class time_block:
    """Context manager for ad-hoc timing blocks."""

    def __init__(
        self,
        name: str,
        *,
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
        logger_: Optional[logging.Logger] = None,
        level: int = logging.DEBUG,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.emit = emit
        self.logger = logger_ or logger
        self.level = level
        self.enabled = enabled
        self._start: float | None = None

    def __enter__(self) -> "time_block":  # noqa: D401
        if self.enabled:
            self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: D401
        if not self.enabled or self._start is None:
            return False
        duration_ms = round((time.time() - self._start) * 1000, 3)
        payload = {"metric": "timing", "name": self.name, "duration_ms": duration_ms, "exception": bool(exc_type)}
        if self.emit:
            try:
                self.emit(payload)
            except Exception:  # pragma: no cover
                self.logger.debug("timing emit callback failed", exc_info=True)
        else:
            if self.logger.isEnabledFor(self.level):
                self.logger.log(self.level, "TIMING %s %sms", self.name, duration_ms)
        return False


__all__ = ["timing", "time_block"]
