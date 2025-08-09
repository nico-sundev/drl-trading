"""
Integration tests for T005 logging standardization.

Tests end-to-end logging functionality including ServiceLogger configuration,
context management, and real log output verification.
"""

import json
import logging
import os
from io import StringIO
from typing import Any
from unittest.mock import patch
import pytest

from drl_trading_common.logging.service_logger import ServiceLogger
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig


class TestServiceLoggerIntegrationDevelopment:
    """Integration tests for ServiceLogger in development environment."""

    def test_complete_development_logging_flow(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test complete logging flow in development environment."""
        # Given
        service_logger = ServiceLogger("integration-test-service", "local")
        service_logger.configure()
        logger = service_logger.get_logger()

        # Create test handler to capture output
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter from existing handler
        console_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                break

        if console_handler and console_handler.formatter:
            test_handler.setFormatter(console_handler.formatter)

        logger.addHandler(test_handler)

        try:
            # When
            logger.info("Development test message")

            # Then
            output = string_stream.getvalue()
            assert "integration-test-service" in output
            assert "Development test message" in output
            assert "INFO" in output
        finally:
            logger.removeHandler(test_handler)

    def test_development_context_manager_integration(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test context manager integration in development."""
        # Given
        service_logger = ServiceLogger("integration-test-service", "local")
        service_logger.configure()
        logger = service_logger.get_logger()

        # Create test handler
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter from existing handler
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.formatter:
                test_handler.setFormatter(handler.formatter)
                break

        logger.addHandler(test_handler)

        try:
            # When
            with service_logger.market_data_context("EURUSD") as _:
                logger.info("Market data processing")

            # Then
            output = string_stream.getvalue()
            assert "Market data processing" in output
            assert "[correlation_id=" in output
            assert "symbol=EURUSD" in output
        finally:
            logger.removeHandler(test_handler)


class TestServiceLoggerIntegrationProduction:
    """Integration tests for ServiceLogger in production environment."""

    def test_complete_production_logging_flow(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test complete logging flow in production environment."""
        # Given
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "production"}):
            service_logger = ServiceLogger("integration-test-service", "prod")
            service_logger.configure()
            logger = service_logger.get_logger()

        # Create test handler to capture JSON output
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter from existing handler
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.formatter:
                test_handler.setFormatter(handler.formatter)
                break

        logger.addHandler(test_handler)

        try:
            # When
            logger.info("Production test message", extra={"operation": "test", "price": 1.2345})

            # Then
            output = string_stream.getvalue()

            # Should be valid JSON
            lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
            for line in lines:
                if "Production test message" in line:
                    log_data = json.loads(line)
                    assert log_data["service"] == "integration-test-service"
                    assert log_data["environment"] == "production"
                    assert log_data["message"] == "Production test message"
                    assert log_data["level"] == "INFO"
                    assert log_data["extra_operation"] == "test"
                    assert log_data["extra_price"] == 1.2345
                    break
            else:
                pytest.fail("Expected log message not found in output")
        finally:
            logger.removeHandler(test_handler)

    def test_production_trading_context_integration(self, temp_directory: str, clean_log_context: Any, sample_trading_context: Any) -> None:
        """Test trading context integration in production."""
        # Given
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "production"}):
            service_logger = ServiceLogger("integration-test-service", "prod")
        service_logger.configure()
        logger = service_logger.get_logger()

        # Create test handler
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter from existing handler
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.formatter:
                test_handler.setFormatter(handler.formatter)
                break

        logger.addHandler(test_handler)

        try:
            # When
            with service_logger.trading_context(sample_trading_context):
                logger.warning("Trading context warning")

            # Then
            output = string_stream.getvalue()
            lines = [line.strip() for line in output.strip().split('\n') if line.strip()]

            for line in lines:
                if "Trading context warning" in line:
                    log_data = json.loads(line)
                    assert log_data["message"] == "Trading context warning"
                    assert log_data["level"] == "WARNING"
                    assert log_data["correlation_id"] == sample_trading_context.correlation_id
                    assert log_data["symbol"] == sample_trading_context.symbol
                    assert log_data["trace_id"] == sample_trading_context.correlation_id  # OpenTelemetry mapping
                    break
            else:
                pytest.fail("Expected log message not found in output")
        finally:
            logger.removeHandler(test_handler)


class TestServiceLoggerFileHandling:
    """Integration tests for file handling functionality."""

    def test_log_file_creation(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test automatic log file creation."""
        # Given
        config = ServiceLoggingConfig(file_logging=True)
        service_logger = ServiceLogger("file-test-service", "local", config)

        # When
        service_logger.configure()
        logger = service_logger.get_logger()
        logger.info("Test log file creation")

        # Flush all handlers to ensure write
        for handler in logger.handlers:
            handler.flush()

        # Then
        log_file_path = "logs/file-test-service.log"
        assert os.path.exists(log_file_path)

        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test log file creation" in content

    def test_production_error_file_creation(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test error file creation in production environment."""
        # Given
        config = ServiceLoggingConfig(file_logging=True)
        service_logger = ServiceLogger("error-test-service", "prod", config)
        service_logger.configure()
        logger = service_logger.get_logger()

        # When
        logger.error("Test error message")

        # Flush all handlers
        for handler in logger.handlers:
            handler.flush()

        # Then
        error_file_path = "logs/error-test-service_errors.log"
        assert os.path.exists(error_file_path)

        with open(error_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Should contain JSON formatted error
            assert "Test error message" in content


class TestServiceLoggerContextManagers:
    """Integration tests for ServiceLogger context managers."""

    def test_correlation_context_manager_integration(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test correlation context manager end-to-end."""
        # Given
        service_logger = ServiceLogger("context-test-service", "local")
        service_logger.configure()
        logger = service_logger.get_logger()

        # Create test handler
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.formatter:
                test_handler.setFormatter(handler.formatter)
                break

        logger.addHandler(test_handler)

        try:
            # When
            with service_logger.correlation_context("test-correlation-integration") as corr_id:
                logger.info("Inside correlation context")
                assert corr_id == "test-correlation-integration"

            # Outside context
            logger.info("Outside correlation context")

            # Then
            output = string_stream.getvalue()
            lines = output.split('\n')

            # First message should have correlation ID
            inside_line = next(line for line in lines if "Inside correlation context" in line)
            assert "[correlation_id=test-correlation-integration]" in inside_line

            # Second message should not have correlation ID
            outside_line = next(line for line in lines if "Outside correlation context" in line)
            assert "correlation_id=" not in outside_line

        finally:
            logger.removeHandler(test_handler)

    def test_nested_context_managers(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test nested context managers."""
        # Given
        service_logger = ServiceLogger("nested-test-service", "local")
        service_logger.configure()
        logger = service_logger.get_logger()

        # Create test handler
        string_stream = StringIO()
        test_handler = logging.StreamHandler(string_stream)

        # Get formatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.formatter:
                test_handler.setFormatter(handler.formatter)
                break

        logger.addHandler(test_handler)

        try:
            # When
            with service_logger.correlation_context("outer-correlation"):
                logger.info("Outer context")

                with service_logger.market_data_context("GBPUSD"):
                    logger.info("Nested context")

            # Then
            output = string_stream.getvalue()
            lines = output.split('\n')

            outer_line = next(line for line in lines if "Outer context" in line)
            assert "[correlation_id=outer-correlation]" in outer_line

            nested_line = next(line for line in lines if "Nested context" in line)
            assert "symbol=GBPUSD" in nested_line
            # Should have a new correlation ID from market_data_context
            assert "correlation_id=" in nested_line

        finally:
            logger.removeHandler(test_handler)


class TestServiceLoggerConfiguration:
    """Integration tests for ServiceLogger configuration scenarios."""

    def test_custom_configuration_integration(self, temp_directory: str, clean_log_context: Any, service_logger_config: Any) -> None:
        """Test custom configuration integration."""
        # Given
        service_logger = ServiceLogger("config-test-service", "local", service_logger_config)
        service_logger.configure()
        logger = service_logger.get_logger()

        # When
        logger.info("Custom configuration test")

        # Then
        # Verify log file is created with custom settings
        log_file_path = "logs/config-test-service.log"
        assert os.path.exists(log_file_path)

        # Verify third-party logger configuration
        urllib3_logger = logging.getLogger('urllib3')
        assert urllib3_logger.level >= logging.WARNING

    def test_environment_switching_integration(self, temp_directory: str, clean_log_context: Any) -> None:
        """Test switching between environments."""
        # Given/When/Then for development
        dev_service_logger = ServiceLogger("env-switch-service", "local")
        dev_service_logger.configure()

        # Verify development setup
        assert dev_service_logger.environment == "development"
        assert not dev_service_logger._should_use_json_formatting()

        # Given/When/Then for production
        prod_service_logger = ServiceLogger("env-switch-service", "prod")
        prod_service_logger.configure()

        # Verify production setup
        assert prod_service_logger.environment == "production"
        assert prod_service_logger._should_use_json_formatting()
