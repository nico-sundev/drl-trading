"""
Structured logging formatters for DRL trading services.

Provides JSON and human-readable formatters that automatically include
trading context information in log entries. Supports both development
and production logging requirements.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Set

from drl_trading_common.logging.trading_log_context import TradingLogContext


class TradingStructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production environments.

    This formatter creates JSON log entries that are compatible with
    log aggregation systems (ELK, Splunk, DataDog) and includes
    trading-specific context information automatically.
    """

    def __init__(self, service_name: str, environment: str = "development"):
        """
        Initialize the structured formatter.

        Args:
            service_name: Name of the service (e.g., 'drl-trading-ingest')
            environment: Deployment environment (development/staging/production)
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = self._get_hostname()

        # Fields to exclude when processing extra record attributes
        self.excluded_fields: Set[str] = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'exc_info', 'exc_text', 'stack_info'
        }

    def _get_hostname(self) -> str:
        """Get hostname, with fallback for different environments."""
        if hasattr(os, 'uname'):
            try:
                return str(os.uname().nodename)  # type: ignore[no-untyped-call]
            except Exception:  # pragma: no cover - defensive
                return os.environ.get('HOSTNAME', 'unknown')
        return os.environ.get('HOSTNAME', 'unknown')

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON with trading context.

        Args:
            record: Python logging record

        Returns:
            JSON string with structured log entry
        """
        # Build base log entry
        short_logger = getattr(record, 'short_name', record.name)
        log_entry: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'service': self.service_name,
            'environment': self.environment,
            'hostname': self.hostname,
            'level': record.levelname,
            'logger': record.name,
            'short_logger': short_logger,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Add trading context information
        trading_context = TradingLogContext.get_available_context()
        log_entry.update(trading_context)

        # Add exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type is not None:
                log_entry['exception'] = {
                    'type': exc_type.__name__,
                    'message': str(exc_value),
                    'traceback': self.formatException(record.exc_info)
                }

        # Add extra fields from log record (prefixed with 'extra_')
        for key, value in record.__dict__.items():
            if key not in self.excluded_fields:
                log_entry[f'extra_{key}'] = value

        # Prepare for OTel compatibility (future integration)
        if trading_context.get('correlation_id'):
            log_entry['trace_id'] = trading_context['correlation_id']  # Future OTel mapping

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class TradingHumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development environments.

    This formatter creates easy-to-read log entries for developers while
    still including essential trading context information.
    """

    MAX_SHORT_NAME_LEN = 40

    def __init__(self, service_name: str):
        """
        Initialize the human-readable formatter.

        Args:
            service_name: Name of the service (e.g., 'drl-trading-ingest')
        """
        self.service_name = service_name

        # Create format string with service name
        format_string = (
            f"%(asctime)s | %(levelname)-8s | %(short_name)-{self.MAX_SHORT_NAME_LEN}s | "
            f"{service_name} | %(message)s"
        )

        super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record for human readability with trading context.

        Args:
            record: Python logging record

        Returns:
            Human-readable log string with context
        """
        # Get base formatted message
        # Ensure graceful fallback if short_name missing (external early logs)
        if not hasattr(record, 'short_name'):
            record.short_name = record.name  # type: ignore[attr-defined]

        # Enforce fixed-width short_name column by truncating with ellipsis when necessary
        original_short = record.short_name  # type: ignore[attr-defined]
        try:
            record.short_name = self._truncate_short_name(original_short)  # type: ignore[attr-defined]
            formatted = super().format(record)
        finally:
            # Restore original to avoid affecting other handlers
            record.short_name = original_short  # type: ignore[attr-defined]

        # Add trading context information if available
        trading_context = TradingLogContext.get_available_context()
        if trading_context:
            context_parts = []

            # Add key context fields in readable format
            for field in ['correlation_id', 'symbol', 'strategy_id', 'model_version']:
                if field in trading_context:
                    context_parts.append(f"{field}={trading_context[field]}")

            if context_parts:
                formatted += f" [{', '.join(context_parts)}]"

        # Add exception information if present
        if record.exc_info:
            et, ev, _ = record.exc_info
            if et is not None:
                formatted += f"\nException: {et.__name__}: {ev}"

        return formatted

    # Internal helpers -------------------------------------------------
    def _truncate_short_name(self, value: str) -> str:
        """Truncate the short_name to MAX_SHORT_NAME_LEN preserving width.

        Uses '...' ellipsis if truncation occurs. Guarantees returned
        string length <= MAX_SHORT_NAME_LEN and pads handled by Formatter.
        """
        max_len = self.MAX_SHORT_NAME_LEN
        if len(value) <= max_len:
            return value
        if max_len <= 3:  # defensive fallback
            return value[:max_len]
        return value[: max_len - 3] + '...'
