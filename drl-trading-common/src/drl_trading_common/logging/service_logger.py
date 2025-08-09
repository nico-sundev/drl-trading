"""
Standardized logging configuration for DRL trading services.

Provides the ServiceLogger class that implements unified logging patterns
across all trading microservices with support for structured JSON logging,
human-readable development logs, and trading-specific context tracking.
"""

import logging
import logging.config
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional
import uuid

from drl_trading_common.logging.trading_formatters import (
    TradingStructuredFormatter,
    TradingHumanReadableFormatter
)
from drl_trading_common.logging.trading_log_context import TradingLogContext
from drl_trading_common.model.trading_context import TradingContext
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig


class ServiceLogger:
    """
    Standardized logger configuration for all DRL trading services.

    This class provides unified logging patterns with:
    - Environment-aware formatting (JSON for production, human-readable for development)
    - Automatic trading context inclusion in all log entries
    - Log rotation and file management
    - Preparation for future OpenTelemetry integration
    """

    def __init__(
        self,
        service_name: str,
        stage: str,
        config: Optional[ServiceLoggingConfig] = None
    ):
        """
        Initialize ServiceLogger for a specific service.

        Args:
            service_name: Name of the service (e.g., 'drl-trading-ingest')
            stage: Deployment stage ('local', 'cicd', 'prod')
            config: Optional ServiceLoggingConfig instance (uses defaults if not provided)
        """
        self.service_name = service_name
        self.stage = stage
        self.config = config or ServiceLoggingConfig()

        # Store reference for context management
        self._logger_instance: Optional[logging.Logger] = None

    @property
    def environment(self) -> str:
        """
        Get environment name for logging context.

        Maps stage to environment names for backwards compatibility.
        """
        environment_mapping = {
            'local': 'development',
            'cicd': 'staging',
            'prod': 'production'
        }
        return environment_mapping.get(self.stage, 'development')

    @property
    def is_production_environment(self) -> bool:
        """Check if running in production-like environment."""
        return self.stage in ['prod', 'cicd']

    def configure(self) -> None:
        """
        Configure logging for the service.

        Sets up formatters, handlers, and loggers based on environment
        and configuration settings.
        """
        # Create logs directory first (before configuring file handlers)
        self._ensure_logs_directory()

        # Determine configuration settings
        log_level = self._get_log_level()
        use_json = self._should_use_json_formatting()

        # Configure formatters
        formatters = self._configure_formatters(use_json)

        # Configure handlers
        handlers = self._configure_handlers(use_json, log_level)

        # Configure loggers
        loggers = self._configure_loggers(log_level)

        # Apply logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': loggers,
            'root': {
                'level': 'WARNING',
                'handlers': ['console']
            }
        }

        logging.config.dictConfig(logging_config)

        # Get service logger and log configuration success
        self._logger_instance = logging.getLogger(self.service_name)
        self._logger_instance.info(f"Logging configured for {self.service_name} service", extra={
            'stage': self.stage,
            'environment': self.environment,
            'json_format': use_json,
            'log_level': log_level
        })

    def _get_log_level(self) -> str:
        """Determine log level from config."""
        return self.config.level

    def _should_use_json_formatting(self) -> bool:
        """Determine if JSON formatting should be used."""
        return self.is_production_environment or self.config.json_format

    def _configure_formatters(self, use_json: bool) -> Dict[str, Any]:
        """Configure logging formatters based on environment."""
        formatters = {}

        if use_json:
            formatters['structured'] = {
                '()': TradingStructuredFormatter,
                'service_name': self.service_name,
                'environment': self.environment
            }
        else:
            formatters['human'] = {
                '()': TradingHumanReadableFormatter,
                'service_name': self.service_name
            }

        return formatters

    def _configure_handlers(self, use_json: bool, log_level: str) -> Dict[str, Any]:
        """Configure logging handlers."""
        formatter_name = 'structured' if use_json else 'human'

        handlers = {}

        # Console handler (always enabled unless explicitly disabled)
        if self.config.console_enabled:
            handlers['console'] = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': formatter_name,
                'stream': 'ext://sys.stdout'
            }

        # Add file handler if enabled
        if self.config.file_logging:
            log_file = self.config.file_path or f'logs/{self.service_name}.log'
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': formatter_name,
                'filename': log_file,
                'maxBytes': self.config.max_file_size,
                'backupCount': self.config.backup_count,
                'mode': 'a'
            }

        # Add error file handler for production-like environments
        if self.is_production_environment:
            error_file = f'logs/{self.service_name}_errors.log'
            handlers['error_file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': formatter_name,
                'filename': error_file,
                'maxBytes': self.config.max_file_size,
                'backupCount': self.config.backup_count,
                'mode': 'a'
            }

        return handlers

    def _configure_loggers(self, log_level: str) -> Dict[str, Any]:
        """Configure service-specific loggers."""
        handler_names = []
        if self.config.console_enabled:
            handler_names.append('console')
        if self.config.file_logging:
            handler_names.append('file')
        if self.is_production_environment:
            handler_names.append('error_file')

        loggers = {
            # Service-specific logger
            self.service_name: {
                'level': log_level,
                'handlers': handler_names,
                'propagate': False
            },
            # Framework loggers
            'drl_trading_common': {
                'level': log_level,
                'handlers': handler_names,
                'propagate': False
            },
            'drl_trading_core': {
                'level': log_level,
                'handlers': handler_names,
                'propagate': False
            }
        }

        # Add third-party logger configurations
        for logger_name, logger_config in self.config.third_party_loggers.items():
            loggers[logger_name] = {
                'level': logger_config.get('level', 'WARNING'),
                'handlers': handler_names,
                'propagate': False
            }

        return loggers

    def _ensure_logs_directory(self) -> None:
        """Create logs directory if it doesn't exist."""
        os.makedirs('logs', exist_ok=True)

    # Context management methods
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """
        Context manager for correlation ID tracking.

        Args:
            correlation_id: Optional correlation ID (generates one if not provided)

        Yields:
            The correlation ID being used
        """
        if correlation_id is None:
            correlation_id = f"trade-{uuid.uuid4()}"

        TradingLogContext.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            TradingLogContext.clear()

    @contextmanager
    def trading_context(self, trading_context: TradingContext):
        """
        Context manager for complete trading context.

        Args:
            trading_context: TradingContext instance with context information
        """
        TradingLogContext.from_trading_context(trading_context)
        try:
            yield
        finally:
            TradingLogContext.clear()

    @contextmanager
    def market_data_context(self, symbol: str, correlation_id: Optional[str] = None):
        """
        Context manager for market data processing.

        Args:
            symbol: Financial instrument symbol
            correlation_id: Optional correlation ID
        """
        if correlation_id is None:
            correlation_id = TradingLogContext.generate_new_correlation_id()
        else:
            TradingLogContext.set_correlation_id(correlation_id)

        TradingLogContext.set_symbol(symbol)
        TradingLogContext.generate_new_event_id(self.service_name)

        try:
            yield
        finally:
            TradingLogContext.clear()

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance for this service.

        Returns:
            Logger instance for the service
        """
        if self._logger_instance is None:
            self._logger_instance = logging.getLogger(self.service_name)
        return self._logger_instance
