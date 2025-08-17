"""
Unit tests for ServiceLogger.

Tests the core ServiceLogger functionality including configuration,
environment-aware formatting, and logger management.
"""

import os
import logging
from typing import Any
import pytest

from drl_trading_common.config.service_logging_config import ServiceLoggingConfig
from drl_trading_common.logging.service_logger import ServiceLogger


class TestServiceLoggerConfiguration:
    """Test ServiceLogger configuration functionality."""

    def test_service_logger_initialization(self, clean_log_context: Any) -> None:
        """Test ServiceLogger basic initialization."""
        # Given
        service_name = "test-service"
        stage = "local"
        config = ServiceLoggingConfig(level="INFO")

        # When
        service_logger = ServiceLogger(service_name, stage, config)

        # Then
        assert service_logger.service_name == service_name
        assert service_logger.stage == stage
        assert service_logger.config == config
        assert service_logger.environment == "development"  # mapped from local
        assert service_logger.is_production_environment is False
        assert service_logger._logger_instance is None

    def test_service_logger_environment_detection(self, clean_log_context: Any) -> None:
        """Test stage to environment mapping."""
        # Given
        stage = "prod"

        # When
        service_logger = ServiceLogger("test-service", stage)

        # Then
        assert service_logger.stage == "prod"
        assert service_logger.environment == "production"
        assert service_logger.is_production_environment is True

    @pytest.mark.parametrize(
        "log_level,expected",
        [
            ("DEBUG", "DEBUG"),
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
        ],
    )
    def test_log_level_configuration(
    self, clean_log_context: Any, log_level: str, expected: str
    ) -> None:
        """Test log level configuration from config and environment."""
        # Given
        config = ServiceLoggingConfig(level=log_level)
        service_logger = ServiceLogger("test-service", "local", config)

        # When
        actual_level = service_logger._get_log_level()

        # Then
        assert actual_level == expected

    def test_log_level_from_config(self, clean_log_context: Any) -> None:
        """Test log level configuration through ServiceLoggingConfig."""
        # Given
        config = ServiceLoggingConfig(level="WARNING")
        service_logger = ServiceLogger("test-service", "local", config)

        # When
        log_level = service_logger._get_log_level()

        # Then
        assert log_level == "WARNING"

    def test_json_formatting_detection_development(self, clean_log_context: Any) -> None:
        """Test JSON formatting detection in development environment."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        should_use_json = service_logger._should_use_json_formatting()

        # Then
        assert should_use_json is False

    def test_json_formatting_detection_production(self, clean_log_context: Any) -> None:
        """Test JSON formatting detection in production environment."""
        # Given
        service_logger = ServiceLogger("test-service", "prod")

        # When
        should_use_json = service_logger._should_use_json_formatting()

        # Then
        assert should_use_json is True

    def test_json_formatting_override(self, clean_log_context: Any) -> None:
        """Test JSON formatting override via configuration."""
        # Given
        config = ServiceLoggingConfig(json_format=True)
        service_logger = ServiceLogger("test-service", "local", config)

        # When
        should_use_json = service_logger._should_use_json_formatting()

        # Then
        assert should_use_json is True


class TestServiceLoggerFormatters:
    """Test ServiceLogger formatter configuration."""

    def test_configure_human_readable_formatter(self, clean_log_context: Any) -> None:
        """Test human-readable formatter configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        formatters = service_logger._configure_formatters(use_json=False)

        # Then
        assert "human" in formatters
        assert formatters["human"]["()"].__name__ == "TradingHumanReadableFormatter"
        assert formatters["human"]["service_name"] == "test-service"

    def test_configure_json_formatter(self, clean_log_context: Any) -> None:
        """Test JSON structured formatter configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        formatters = service_logger._configure_formatters(use_json=True)

        # Then
        assert "structured" in formatters
        assert formatters["structured"]["()"].__name__ == "TradingStructuredFormatter"
        assert formatters["structured"]["service_name"] == "test-service"
        assert formatters["structured"]["environment"] == "development"


class TestServiceLoggerHandlers:
    """Test ServiceLogger handler configuration."""

    def test_console_handler_configuration(self, clean_log_context: Any) -> None:
        """Test console handler configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        handlers = service_logger._configure_handlers(use_json=False, log_level="INFO")

        # Then
        assert "console" in handlers
        console_handler = handlers["console"]
        assert console_handler["class"] == "logging.StreamHandler"
        assert console_handler["level"] == "INFO"
        assert console_handler["formatter"] == "human"

    def test_file_handler_configuration(
    self, clean_log_context: Any, service_logger_config: ServiceLoggingConfig
    ) -> None:
        """Test file handler configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local", service_logger_config)

        # When
        handlers = service_logger._configure_handlers(use_json=False, log_level="INFO")

        # Then
        assert "file" in handlers
        file_handler = handlers["file"]
        assert file_handler["class"] == "logging.handlers.RotatingFileHandler"
        assert file_handler["filename"] == "logs/test-service.log"
        assert file_handler["maxBytes"] == 1024 * 1024  # From config
        assert file_handler["backupCount"] == 3  # From config

    def test_error_file_handler_production(self, clean_log_context: Any) -> None:
        """Test error file handler configuration in production."""
        # Given
        service_logger = ServiceLogger("test-service", "prod")

        # When
        handlers = service_logger._configure_handlers(use_json=True, log_level="INFO")

        # Then
        assert "error_file" in handlers
        error_handler = handlers["error_file"]
        assert error_handler["class"] == "logging.handlers.RotatingFileHandler"
        assert error_handler["level"] == "ERROR"
        assert error_handler["filename"] == "logs/test-service_errors.log"

    def test_no_error_file_handler_development(self, clean_log_context: Any) -> None:
        """Test no error file handler in development."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        handlers = service_logger._configure_handlers(use_json=False, log_level="INFO")

        # Then
        assert "error_file" not in handlers


class TestServiceLoggerLoggers:
    """Test ServiceLogger logger configuration."""

    def test_service_logger_configuration(self, clean_log_context: Any) -> None:
        """Test service-specific logger configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        loggers = service_logger._configure_loggers("INFO")

        # Then
        assert "test-service" in loggers
        service_config = loggers["test-service"]
        assert service_config["level"] == "INFO"
        assert "console" in service_config["handlers"]
        assert service_config["propagate"] is False

    def test_framework_loggers_configuration(self, clean_log_context: Any) -> None:
        """Test framework loggers configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        loggers = service_logger._configure_loggers("INFO")

        # Then
        assert "drl_trading_common" in loggers
        assert "drl_trading_core" in loggers

        for logger_name in ["drl_trading_common", "drl_trading_core"]:
            logger_config = loggers[logger_name]
            assert logger_config["level"] == "INFO"
            assert "console" in logger_config["handlers"]
            assert logger_config["propagate"] is False

    def test_third_party_loggers_configuration(
    self, clean_log_context: Any, service_logger_config: ServiceLoggingConfig
    ) -> None:
        """Test third-party loggers configuration."""
        # Given
        service_logger = ServiceLogger("test-service", "local", service_logger_config)

        # When
        loggers = service_logger._configure_loggers("INFO")

        # Then
        assert "urllib3" in loggers
        assert "requests" in loggers

        assert loggers["urllib3"]["level"] == "WARNING"
        assert loggers["requests"]["level"] == "WARNING"


class TestServiceLoggerIntegration:
    """Test ServiceLogger end-to-end integration."""

    def test_full_configuration(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test complete ServiceLogger configuration process."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        service_logger.configure()
        logger = service_logger.get_logger()

        # Then
        assert logger is not None
        assert logger.name == "test-service"
        assert service_logger._logger_instance is logger

    def test_logs_directory_creation(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test automatic logs directory creation."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        service_logger.configure()

        # Then
        assert os.path.exists("logs")
        assert os.path.isdir("logs")

    def test_get_logger_before_configure(self, clean_log_context: Any) -> None:
        """Test get_logger before configure() is called."""
        # Given
        service_logger = ServiceLogger("test-service", "local")

        # When
        logger = service_logger.get_logger()

        # Then
        assert logger is not None
        assert logger.name == "test-service"

    def test_get_logger_caching(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test logger instance caching."""
        # Given
        service_logger = ServiceLogger("test-service", "local")
        service_logger.configure()

        # When
        logger1 = service_logger.get_logger()
        logger2 = service_logger.get_logger()

        # Then
        assert logger1 is logger2


class TestServiceLoggerShortName:
    """Tests for short_name abbreviation feature in ServiceLogger."""
    def test_short_name_abbreviation_deep_logger(self, clean_log_context: Any) -> None:
        """Ensure deep dotted logger names are abbreviated correctly.

        Given a deep module path > 3 segments
        When a log record is emitted after ServiceLogger.configure()
        Then record.short_name follows abbreviation rules and is cached.
        """
        # Given
        service_logger = ServiceLogger("test-service", "local")
        service_logger.configure()
        deep_logger_name = "drl_trading_preprocess.adapter.messaging.kafka.consumer.SomeComponent"
        deep_logger = logging.getLogger(deep_logger_name)

        class CaptureHandler(logging.Handler):  # simple in-memory record collector
            def __init__(self) -> None:  # type: ignore[no-untyped-def]
                super().__init__()
                self.records: list[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                self.records.append(record)

        handler = CaptureHandler()
        deep_logger.addHandler(handler)
        deep_logger.setLevel(logging.INFO)
        deep_logger.propagate = False

        # When
        deep_logger.info("Deep logger test message")

        # Then
        assert len(handler.records) == 1
        record = handler.records[0]
        # Expected abbreviation under implemented simplified rule: first letter of each segment except final kept full
        # drl_trading_preprocess.adapter.messaging.kafka.consumer.SomeComponent -> d.a.m.k.c.SomeComponent
        assert getattr(record, "short_name", None) == "d.a.m.k.c.SomeComponent"
        # Cache should contain original name
        assert ServiceLogger._SHORT_NAME_CACHE[deep_logger_name] == "d.a.m.k.c.SomeComponent"

    def test_short_name_no_abbreviation_shallow_logger(self, clean_log_context: Any) -> None:
        """Shallow logger names (<=3 segments) should remain unchanged."""
        # Given
        service_logger = ServiceLogger("test-service", "local")
        service_logger.configure()
        shallow_name = "drl_trading_common"  # single segment => unchanged
        logger_obj = logging.getLogger(shallow_name)

        class CaptureHandler(logging.Handler):
            def __init__(self) -> None:  # type: ignore[no-untyped-def]
                super().__init__()
                self.records: list[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                self.records.append(record)

        handler = CaptureHandler()
        logger_obj.addHandler(handler)
        logger_obj.setLevel(logging.INFO)
        logger_obj.propagate = False

        # When
        logger_obj.info("Shallow logger message")

        # Then
        assert len(handler.records) == 1
        record = handler.records[0]
        assert getattr(record, "short_name", None) == shallow_name

    def test_record_factory_idempotent(self, clean_log_context: Any) -> None:
        """Calling configure multiple times should not reinstall record factory."""
        # Given
        service_logger = ServiceLogger("test-service", "local")
        service_logger.configure()
        initial_flag = ServiceLogger._record_factory_installed
        assert initial_flag is True

        # When
        service_logger.configure()  # second invocation

        # Then
        assert ServiceLogger._record_factory_installed is True  # still installed
        # Abbreviation still works after second configure
        deep_name = "drl_trading_preprocess.adapter.messaging.kafka.consumer.SomeComponent"
        logger_obj = logging.getLogger(deep_name)
        logger_obj.info("Idempotent test")
        # Confirm cache entry created without error
        assert deep_name in ServiceLogger._SHORT_NAME_CACHE

    def test_json_formatter_includes_short_logger(self, clean_log_context: Any) -> None:
        """JSON structured output should include short_logger field when enabled."""
        # Given
        config = ServiceLoggingConfig(json_format=True, abbreviate_logger_names=True)
        service_logger = ServiceLogger("test-service", "local", config)
        service_logger.configure()
        deep_logger_name = "drl_trading_preprocess.adapter.messaging.kafka.consumer.SomeComponent"
        logger_obj = logging.getLogger(deep_logger_name)

        class CaptureHandler(logging.Handler):
            def __init__(self) -> None:  # type: ignore[no-untyped-def]
                super().__init__()
                self.records: list[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                self.records.append(record)

        # Attach a structured formatter manually to capture JSON string
        capture = CaptureHandler()
        from drl_trading_common.logging.trading_formatters import TradingStructuredFormatter
        capture.setFormatter(TradingStructuredFormatter(service_name="test-service", environment="development"))
        logger_obj.addHandler(capture)
        logger_obj.setLevel(logging.INFO)
        logger_obj.propagate = False

        # When
        logger_obj.info("Structured abbreviation test")

        # Then
        assert len(capture.records) == 1
        output = capture.format(capture.records[0])
        assert '"short_logger"' in output
        assert 'consumer.SomeComponent' in output
