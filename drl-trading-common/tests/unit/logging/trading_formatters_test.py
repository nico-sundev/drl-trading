"""
Unit tests for TradingFormatters.

Tests the TradingStructuredFormatter and TradingHumanReadableFormatter
functionality including context integration and formatting.
"""

import json
import logging
from datetime import datetime
from typing import Any

from drl_trading_common.logging.trading_formatters import (
    TradingStructuredFormatter,
    TradingHumanReadableFormatter
)
from drl_trading_common.logging.trading_log_context import TradingLogContext


class TestTradingStructuredFormatter:
    """Test TradingStructuredFormatter for JSON logging."""

    def test_formatter_initialization(self, clean_log_context: Any) -> None:
        """Test structured formatter initialization."""
        # Given
        service_name = "test-service"
        environment = "production"

        # When
        formatter = TradingStructuredFormatter(service_name, environment)

        # Then
        assert formatter.service_name == service_name
        assert formatter.environment == environment

    def test_basic_log_formatting(self, clean_log_context: Any) -> None:
        """Test basic log record formatting to JSON."""
        # Given
        formatter = TradingStructuredFormatter("test-service", "production")
        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Then
        assert parsed["service"] == "test-service"
        assert parsed["environment"] == "production"
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42
        assert "timestamp" in parsed
        assert "hostname" in parsed
        assert parsed["short_logger"] == "test-logger"  # fallback to full name when no short_name

    def test_short_logger_uses_record_short_name(self, clean_log_context: Any) -> None:
        """Structured formatter should emit provided record.short_name."""
        # Given
        formatter = TradingStructuredFormatter("test-service", "production")
        record = logging.LogRecord(
            name="very.deep.module.structure.Class",
            level=logging.INFO,
            pathname="/p.py",
            lineno=1,
            msg="Deep message",
            args=(),
            exc_info=None
        )
        record.short_name = "v.d.m.s.Class"  # Simulate abbreviation injected by ServiceLogger

        # When
        parsed = json.loads(formatter.format(record))

        # Then
        assert parsed["short_logger"] == "v.d.m.s.Class"

    def test_trading_context_integration(self, clean_log_context: Any) -> None:
        """Test integration with TradingLogContext."""
        # Given
        formatter = TradingStructuredFormatter("test-service", "production")
        TradingLogContext.set_correlation_id("test-correlation-789")
        TradingLogContext.set_symbol("GBPJPY")
        TradingLogContext.set_strategy_id("scalping_v3")

        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=100,
            msg="Trading operation",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Then
        assert parsed["correlation_id"] == "test-correlation-789"
        assert parsed["symbol"] == "GBPJPY"
        assert parsed["strategy_id"] == "scalping_v3"
        assert parsed["trace_id"] == "test-correlation-789"  # OpenTelemetry mapping

    def test_extra_fields_integration(self, clean_log_context: Any) -> None:
        """Test extra fields from log record."""
        # Given
        formatter = TradingStructuredFormatter("test-service", "production")
        record = logging.LogRecord(
            name="test-logger",
            level=logging.ERROR,
            pathname="/path/to/module.py",
            lineno=200,
            msg="Operation failed",
            args=(),
            exc_info=None
        )
        # Add extra fields
        record.price = 1.2345
        record.volume = 1000
        record.error_code = "TIMEOUT_ERROR"

        # When
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Then
        assert parsed["extra_price"] == 1.2345
        assert parsed["extra_volume"] == 1000
        assert parsed["extra_error_code"] == "TIMEOUT_ERROR"

    def test_timestamp_formatting(self, clean_log_context: Any) -> None:
        """Test ISO timestamp formatting."""
        # Given
        formatter = TradingStructuredFormatter("test-service", "production")
        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Then
        timestamp_str = parsed["timestamp"]
        # Should be parseable as ISO format
        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        assert parsed_timestamp is not None


class TestTradingHumanReadableFormatter:
    """Test TradingHumanReadableFormatter for development logging."""

    def test_formatter_initialization(self, clean_log_context: Any) -> None:
        """Test human-readable formatter initialization."""
        # Given
        service_name = "test-service"

        # When
        formatter = TradingHumanReadableFormatter(service_name)

        # Then
        assert formatter.service_name == service_name

    def test_basic_log_formatting(self, clean_log_context: Any) -> None:
        """Test basic log record formatting for humans."""
        # Given
        formatter = TradingHumanReadableFormatter("test-service")
        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)

        # Then
        assert "INFO" in formatted
        assert "test-service" in formatted
        assert "Test message" in formatted
        assert "|" in formatted  # Should have separator pipes
        # Ensure logger name (fallback) present
        assert "test-logger" in formatted

    def test_human_formatter_uses_short_name(self, clean_log_context: Any) -> None:
        """If record.short_name provided it should appear instead of full name."""
        # Given
        formatter = TradingHumanReadableFormatter("test-service")
        record = logging.LogRecord(
            name="very.deep.module.structure.Component",
            level=logging.INFO,
            pathname="/p.py",
            lineno=10,
            msg="Abbrev test",
            args=(),
            exc_info=None
        )
        record.short_name = "v.d.m.s.Component"

        # When
        formatted = formatter.format(record)

        # Then
        assert "v.d.m.s.Component" in formatted
        assert "very.deep.module.structure.Component" not in formatted

    def test_trading_context_integration(self, clean_log_context: Any) -> None:
        """Test integration with TradingLogContext."""
        # Given
        formatter = TradingHumanReadableFormatter("test-service")
        TradingLogContext.set_correlation_id("test-correlation-999")
        TradingLogContext.set_symbol("AUDCAD")

        record = logging.LogRecord(
            name="test-logger",
            level=logging.WARNING,
            pathname="/path/to/module.py",
            lineno=150,
            msg="Trading warning",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)

        # Then
        assert "[correlation_id=test-correlation-999" in formatted
        assert "symbol=AUDCAD]" in formatted

    def test_context_formatting_none_values(self, clean_log_context: Any) -> None:
        """Test context formatting excludes None values."""
        # Given
        formatter = TradingHumanReadableFormatter("test-service")
        TradingLogContext.set_correlation_id("test-123")
        # Don't set other context values - they should be None

        record = logging.LogRecord(
            name="test-logger",
            level=logging.DEBUG,
            pathname="/path/to/module.py",
            lineno=75,
            msg="Debug message",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)

        # Then
        assert "[correlation_id=test-123]" in formatted
        # Should not contain None values
        assert "symbol=None" not in formatted
        assert "strategy_id=None" not in formatted

    def test_no_context_formatting(self, clean_log_context: Any) -> None:
        """Test formatting when no trading context is set."""
        # Given
        formatter = TradingHumanReadableFormatter("test-service")
        # No context set

        record = logging.LogRecord(
            name="test-logger",
            level=logging.ERROR,
            pathname="/path/to/module.py",
            lineno=300,
            msg="Error message",
            args=(),
            exc_info=None
        )

        # When
        formatted = formatter.format(record)

        # Then
        assert "ERROR" in formatted
        assert "test-service" in formatted
        assert "Error message" in formatted
        # Should not have empty context brackets
        assert "[]" not in formatted


class TestFormattersComparison:
    """Test comparison between formatters in different environments."""

    def test_development_vs_production_formatting(self, clean_log_context: Any) -> None:
        """Test that formatters produce appropriate output for their environment."""
        # Given
        human_formatter = TradingHumanReadableFormatter("test-service")
        json_formatter = TradingStructuredFormatter("test-service", "production")

        TradingLogContext.set_correlation_id("compare-test-123")
        TradingLogContext.set_symbol("NZDUSD")

        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=500,
            msg="Comparison test",
            args=(),
            exc_info=None
        )

        # When
        human_output = human_formatter.format(record)
        json_output = json_formatter.format(record)

        # Then
        # Human output should be readable
        assert "Comparison test" in human_output
        assert "INFO" in human_output
        assert "[correlation_id=compare-test-123" in human_output

        # JSON output should be parseable
        json_data = json.loads(json_output)
        assert json_data["message"] == "Comparison test"
        assert json_data["level"] == "INFO"
        assert json_data["correlation_id"] == "compare-test-123"
        assert json_data["symbol"] == "NZDUSD"
