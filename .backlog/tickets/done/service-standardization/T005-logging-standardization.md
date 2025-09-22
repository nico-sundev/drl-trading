# T005: Logging Configuration Standardization

**Priority:** High
**Estimated Effort:** 2 days
**Type:** Implementation
**Depends On:** T004

## Objective
Implement unified logging configuration patterns across all DRL trading microservices to ensure consistent log formatting, structured logging for production, and centralized log management compatibility.

## Current State Analysis

### Existing Logging Patterns (Inconsistent):
- **drl-trading-inference**: Custom `setup_logging()` function
- **drl-trading-training**: Basic logging with different formatters
- **drl-trading-ingest**: Flask-specific logging configuration
- **drl-trading-core**: Various logging setups across modules

### Problems with Current Approach:
- **Inconsistent Log Formats**: Different timestamp formats, message structures
- **No Structured Logging**: Missing JSON formatting for production monitoring
- **Missing Context**: No service identification, request IDs, or correlation IDs
- **No Log Levels Configuration**: Hardcoded log levels across services
- **Missing Production Features**: No log rotation, sampling, or filtering
- **Poor Observability**: Difficult to correlate logs across services

## Standardized Logging Design

### 1. Enhanced Logging Framework
**Duration:** 1 day

```python
# drl_trading_common/logging/service_logger.py
import logging
import logging.config
import json
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from threading import local
import uuid

class LogContext:
    """Thread-local storage for logging context."""
    _local = local()

    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(cls._local, 'correlation_id', None)

    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """Set request ID for current thread."""
        cls._local.request_id = request_id

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get request ID for current thread."""
        return getattr(cls._local, 'request_id', None)

    @classmethod
    def set_user_id(cls, user_id: str) -> None:
        """Set user ID for current thread."""
        cls._local.user_id = user_id

    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Get user ID for current thread."""
        return getattr(cls._local, 'user_id', None)

    @classmethod
    def clear(cls) -> None:
        """Clear all context for current thread."""
        for attr in ['correlation_id', 'request_id', 'user_id']:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, service_name: str, environment: str = "development"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'service': self.service_name,
            'environment': self.environment,
            'hostname': self.hostname,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Add context information if available
        if LogContext.get_correlation_id():
            log_entry['correlation_id'] = LogContext.get_correlation_id()
        if LogContext.get_request_id():
            log_entry['request_id'] = LogContext.get_request_id()
        if LogContext.get_user_id():
            log_entry['user_id'] = LogContext.get_user_id()

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno',
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[f'extra_{key}'] = value

        return json.dumps(log_entry, ensure_ascii=False)

class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-30s | "
            f"{service_name} | %(message)s"
        )
        super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        formatted = super().format(record)

        # Add context information if available
        context_parts = []
        if LogContext.get_correlation_id():
            context_parts.append(f"correlation_id={LogContext.get_correlation_id()}")
        if LogContext.get_request_id():
            context_parts.append(f"request_id={LogContext.get_request_id()}")

        if context_parts:
            formatted += f" [{', '.join(context_parts)}]"

        return formatted

class ServiceLogger:
    """Standardized logger configuration for all services."""

    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self.environment = os.environ.get("DEPLOYMENT_MODE", "development")

    def configure(self) -> None:
        """Configure logging for the service."""
        # Determine log level
        log_level = self.config.get('level', os.environ.get('LOG_LEVEL', 'INFO'))

        # Determine output format based on environment
        use_json = self.environment in ['production', 'staging'] or self.config.get('json_format', False)

        # Configure formatters
        formatters = {}
        if use_json:
            formatters['structured'] = {
                '()': StructuredFormatter,
                'service_name': self.service_name,
                'environment': self.environment
            }
        else:
            formatters['human'] = {
                '()': HumanReadableFormatter,
                'service_name': self.service_name
            }

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

        # Create logs directory if needed
        os.makedirs('logs', exist_ok=True)

        # Log configuration success
        logger = logging.getLogger(self.service_name)
        logger.info(f"Logging configured for {self.service_name} service", extra={
            'environment': self.environment,
            'json_format': use_json,
            'log_level': log_level
        })

    def _configure_handlers(self, use_json: bool, log_level: str) -> Dict[str, Any]:
        """Configure logging handlers."""
        formatter_name = 'structured' if use_json else 'human'

        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': formatter_name,
                'stream': 'ext://sys.stdout'
            }
        }

        # Add file handler if configured
        if self.config.get('file_logging', True):
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': formatter_name,
                'filename': f'logs/{self.service_name}.log',
                'maxBytes': self.config.get('max_file_size', 10 * 1024 * 1024),  # 10MB
                'backupCount': self.config.get('backup_count', 5),
                'mode': 'a'
            }

        # Add error file handler for production
        if self.environment in ['production', 'staging']:
            handlers['error_file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': formatter_name,
                'filename': f'logs/{self.service_name}_errors.log',
                'maxBytes': self.config.get('max_file_size', 10 * 1024 * 1024),
                'backupCount': self.config.get('backup_count', 10),
                'mode': 'a'
            }

        return handlers

    def _configure_loggers(self, log_level: str) -> Dict[str, Any]:
        """Configure service-specific loggers."""
        handler_names = ['console']
        if self.config.get('file_logging', True):
            handler_names.append('file')
        if self.environment in ['production', 'staging']:
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
        third_party_loggers = self.config.get('third_party_loggers', {})
        for logger_name, logger_config in third_party_loggers.items():
            loggers[logger_name] = {
                'level': logger_config.get('level', 'WARNING'),
                'handlers': handler_names,
                'propagate': False
            }

        return loggers

    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        LogContext.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            LogContext.clear()

    @contextmanager
    def request_context(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        """Context manager for request context."""
        if request_id:
            LogContext.set_request_id(request_id)
        if user_id:
            LogContext.set_user_id(user_id)

        try:
            yield
        finally:
            LogContext.clear()
```

### 2. Configuration Schema Enhancement
**Duration:** 0.5 days

```python
# drl_trading_common/config/logging_config.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class LoggingConfig(BaseModel):
    """Configuration schema for logging."""

    level: str = Field(default="INFO", description="Default log level")
    json_format: bool = Field(default=False, description="Use JSON formatting")
    file_logging: bool = Field(default=True, description="Enable file logging")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, description="Number of backup log files")

    # Third-party logger configurations
    third_party_loggers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for third-party loggers"
    )

    # Performance settings
    sampling_rate: Optional[float] = Field(
        default=None,
        description="Log sampling rate for high-volume logs (0.0-1.0)"
    )

    # Production-specific settings
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )

    class Config:
        extra = "forbid"

# Add to service configurations
class BaseServiceConfig(BaseApplicationConfig):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

### 3. Integration with Service Bootstrap
**Duration:** 0.5 days

```python
# Update T004 ServiceBootstrap to use ServiceLogger
class ServiceBootstrap(ABC):
    def _setup_logging(self) -> None:
        """Setup standardized logging configuration."""
        service_logger = ServiceLogger(
            service_name=self.service_name,
            config=self.config.logging.dict() if hasattr(self.config, 'logging') else None
        )
        service_logger.configure()

        # Store logger instance for context management
        self.service_logger = service_logger
```

## Production-Ready Features

### 1. Performance Optimization
```python
# drl_trading_common/logging/performance.py
import logging
import random
from typing import Optional

class SamplingFilter(logging.Filter):
    """Filter for log sampling to reduce volume."""

    def __init__(self, sampling_rate: float):
        super().__init__()
        self.sampling_rate = sampling_rate

    def filter(self, record: logging.LogRecord) -> bool:
        """Sample logs based on configured rate."""
        if self.sampling_rate >= 1.0:
            return True
        return random.random() < self.sampling_rate

class HighVolumeLogger:
    """Logger wrapper for high-volume scenarios."""

    def __init__(self, logger: logging.Logger, sampling_rate: Optional[float] = None):
        self.logger = logger
        self.sampling_rate = sampling_rate or 1.0

    def log_market_data(self, symbol: str, price: float, volume: int):
        """Log market data with sampling."""
        if random.random() < self.sampling_rate:
            self.logger.debug("Market data received", extra={
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'event_type': 'market_data'
            })
```

### 2. Error Tracking Integration
```python
# drl_trading_common/logging/error_tracking.py
import logging
from typing import Optional

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

class SentryHandler(logging.Handler):
    """Custom Sentry handler for error tracking."""

    def __init__(self, dsn: str, service_name: str):
        super().__init__(level=logging.ERROR)
        if SENTRY_AVAILABLE:
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            sentry_sdk.init(
                dsn=dsn,
                integrations=[sentry_logging],
                environment=os.environ.get("DEPLOYMENT_MODE", "development"),
                server_name=service_name
            )

    def emit(self, record: logging.LogRecord):
        """Emit log record to Sentry."""
        if SENTRY_AVAILABLE and record.levelno >= logging.ERROR:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("service", "drl_trading")
                scope.set_tag("logger", record.name)

                if LogContext.get_correlation_id():
                    scope.set_tag("correlation_id", LogContext.get_correlation_id())

                if record.exc_info:
                    sentry_sdk.capture_exception(record.exc_info[1])
                else:
                    sentry_sdk.capture_message(record.getMessage(), level=record.levelname.lower())
```

## Testing Strategy

### Unit Tests
```python
class TestServiceLogger:
    def test_json_formatting_production(self):
        """Test JSON formatting in production environment."""
        # Given
        os.environ["DEPLOYMENT_MODE"] = "production"
        service_logger = ServiceLogger("test_service")

        # When
        service_logger.configure()
        logger = logging.getLogger("test_service")

        # Capture log output
        with self.capture_logs() as log_capture:
            logger.info("Test message", extra={'custom_field': 'value'})

        # Then
        log_entry = json.loads(log_capture.getvalue())
        assert log_entry['service'] == 'test_service'
        assert log_entry['level'] == 'INFO'
        assert log_entry['message'] == 'Test message'
        assert log_entry['extra_custom_field'] == 'value'

    def test_correlation_context(self):
        """Test correlation ID context management."""
        # Given
        service_logger = ServiceLogger("test_service")
        service_logger.configure()

        # When
        with service_logger.correlation_context("test-correlation-123"):
            logger = logging.getLogger("test_service")
            with self.capture_logs() as log_capture:
                logger.info("Test message")

        # Then
        if service_logger.environment in ['production', 'staging']:
            log_entry = json.loads(log_capture.getvalue())
            assert log_entry['correlation_id'] == 'test-correlation-123'
```

## Migration Strategy

### Phase 1: Framework Deployment
1. Deploy `ServiceLogger` to `drl-trading-common`
2. Update service configuration schemas
3. Create migration utilities

### Phase 2: Service-by-Service Migration
1. **drl-trading-inference**: Update bootstrap to use `ServiceLogger`
2. **drl-trading-training**: Migrate logging configuration
3. **drl-trading-ingest**: Replace Flask logging with standardized approach
4. **Remaining services**: Apply consistent pattern

### Phase 3: Production Enhancements
1. Enable JSON logging in production
2. Configure log rotation and archival
3. Set up centralized log aggregation
4. Configure error tracking (Sentry)

## Acceptance Criteria
- [ ] All services use identical logging configuration with `ServiceLogger`
- [ ] JSON structured logging enabled for production environments
- [ ] Human-readable logging for development environments
- [ ] Correlation ID and request context tracking implemented
- [ ] Log rotation and file management configured
- [ ] Error tracking integration (Sentry) available
- [ ] Performance optimization (sampling) implemented for high-volume logs
- [ ] All existing log statements maintain their intent and information
- [ ] No log information loss during migration
- [ ] Log aggregation compatibility verified

## Dependencies
- **Depends On:** T004 (Service Bootstrap Patterns)
- **Blocks:** T006 (Service Migration & Validation)

## Risks
- **Log Volume**: Structured logging could increase log volume
  - **Mitigation**: Implement sampling and log level management
- **Performance Impact**: JSON formatting overhead
  - **Mitigation**: Performance testing and optimization
- **Information Loss**: Migration could lose existing log context
  - **Mitigation**: Careful migration with validation

## Definition of Done
- [ ] `ServiceLogger` framework implemented and tested
- [ ] All services migrated to use standardized logging
- [ ] JSON logging working in production environments
- [ ] Context tracking (correlation ID, request ID) functional
- [ ] Log rotation and file management configured
- [ ] Performance impact measured and acceptable
- [ ] Error tracking integration tested
- [ ] Documentation complete with usage examples
- [ ] No regression in log information or quality
