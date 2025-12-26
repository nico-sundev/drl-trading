"""Startup observability utilities shared across services.

Provides a lightweight context object to capture phase timings and
dependency health so every service can emit a consistent structured
startup summary. Kept intentionally framework-light to avoid coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Callable, Iterator
from contextlib import contextmanager
import logging
import json

from drl_trading_common.instrumentation.timing import time_block

logger = logging.getLogger(__name__)


class DependencyHealth(str, Enum):
    """Health states for startup dependencies."""
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    DEGRADED = "DEGRADED"  # Used when non-mandatory failed


@dataclass
class DependencyStatus:
    """Represents the startup status of an external dependency."""
    name: str
    mandatory: bool
    healthy: bool
    message: str = ""
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "name": self.name,
            "mandatory": self.mandatory,
            "healthy": self.healthy,
            "message": self.message,
            "latency_ms": self.latency_ms,
        }


class StartupContext:
    """Accumulates startup metrics for a service.

    Usage:
        ctx = StartupContext(service_name)
        with ctx.phase("config"):
            ...
        ctx.add_dependency_status(...)
        ctx.emit_summary(logger)
    """

    def __init__(
        self,
        service_name: str,
        *,
        emit_phase_metrics: bool = True,
        metric_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.service_name = service_name
        self._start_time = time.time()
        self._phases: Dict[str, float] = {}
        self._phase_starts: Dict[str, float] = {}
        self._dependencies: List[DependencyStatus] = []
        self._attributes: Dict[str, Any] = {}
        self._emit_phase_metrics = emit_phase_metrics
        self._metric_emit = metric_emit
        self._phase_exceptions: Dict[str, str] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        tb: Optional[time_block] = None
        if self._emit_phase_metrics:
            tb = time_block(
                f"startup.phase.{name}",
                emit=self._metric_emit,
                enabled=True,
            )
            tb.__enter__()
        self.start_phase(name)
        try:
            yield
        except BaseException as e:  # noqa: BLE001
            self._phase_exceptions[name] = e.__class__.__name__
            if tb:
                tb.__exit__(type(e), e, e.__traceback__)
            self.end_phase(name)
            raise
        else:
            if tb:
                tb.__exit__(None, None, None)
            self.end_phase(name)

    def start_phase(self, name: str) -> None:
        self._phase_starts[name] = time.time()

    def end_phase(self, name: str) -> None:
        if name in self._phase_starts and name not in self._phases:
            duration = round((time.time() - self._phase_starts[name]) * 1000, 2)
            self._phases[name] = duration
            del self._phase_starts[name]

    def add_dependency_status(
        self,
        name: str,
        healthy: bool,
        mandatory: bool = True,
        message: str = "",
        latency_ms: Optional[float] = None,
    ) -> None:
        self._dependencies.append(
            DependencyStatus(
                name=name,
                healthy=healthy,
                mandatory=mandatory,
                message=message,
                latency_ms=latency_ms,
            )
        )

    def attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def mandatory_dependencies_healthy(self) -> bool:
        return all(d.healthy or (not d.mandatory) for d in self._dependencies)

    def summary_dict(self) -> Dict[str, Any]:
        total_time_ms = round((time.time() - self._start_time) * 1000, 2)
        for pending in list(self._phase_starts.keys()):
            if pending not in self._phases:
                self.end_phase(pending)
        phases_sorted = [
            {"name": name, "duration_ms": duration}
            for name, duration in self._phases.items()
        ]
        dependencies = [d.to_dict() for d in self._dependencies]
        status = "HEALTHY" if self.mandatory_dependencies_healthy() else "DEGRADED"
        return {
            "service": self.service_name,
            "status": status,
            "total_time_ms": total_time_ms,
            "phases": phases_sorted,
            "dependencies": dependencies,
            "attributes": self._attributes,
            "phase_exceptions": self._phase_exceptions,
        }

    def emit_summary(self, log: logging.Logger) -> None:
        try:
            payload = self.summary_dict()
            log.info("STARTUP SUMMARY %s", json.dumps(payload, sort_keys=True))
        except Exception as e:  # pragma: no cover
            log.warning(f"Failed to emit startup summary: {e}")
