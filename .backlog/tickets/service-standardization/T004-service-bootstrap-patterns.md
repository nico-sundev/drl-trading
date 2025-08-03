# T004: Service Bootstrap Pattern Standardization

**Priority:** High
**Estimated Effort:** 3 days
**Type:** Implementation
**Depends On:** T002, T003

## Objective
Establish uniform service bootstrap patterns across all DRL trading microservices, including standardized dependency injection container initialization, logging setup, configuration loading, and graceful shutdown handling.

## Current State Analysis

### Existing Patterns (Inconsistent):

#### drl-trading-inference/main.py
```python
from drl_trading_inference import setup_logging, bootstrap_inference_service

if __name__ == "__main__":
    setup_logging()
    bootstrap_inference_service()
```

#### drl-trading-training/main.py
```python
from .src.drl_trading_training import TrainingApp

if __name__ == "__main__":
    app = TrainingApp()
    app.run()
```

#### drl-trading-ingest/main.py
```python
import logging
from flask import Flask
from injector import Injector
from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule

def create_app():
    # Flask-specific bootstrap logic
```

### Problems with Current Approach:
- **Inconsistent Entry Points**: Different patterns across services
- **No Standard DI Initialization**: Each service has custom DI setup
- **Inconsistent Logging**: Different logging configuration approaches
- **No Graceful Shutdown**: Services don't handle SIGTERM properly
- **No Health Checks**: Missing standardized health/readiness endpoints
- **Configuration Loading**: Different configuration loading patterns

## Standardized Bootstrap Design

### 1. Common Bootstrap Interface
**Duration:** 1 day

```python
# drl_trading_common/bootstrap/service_bootstrap.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import signal
import logging
import sys
from injector import Injector

logger = logging.getLogger(__name__)

class ServiceBootstrap(ABC):
    """
    Abstract base class for standardized service bootstrapping.

    Provides common patterns for:
    - Configuration loading
    - Dependency injection setup
    - Logging configuration
    - Graceful shutdown handling
    - Health check endpoints
    """

    def __init__(self, service_name: str, config_class: Type[BaseApplicationConfig]):
        self.service_name = service_name
        self.config_class = config_class
        self.config: Optional[BaseApplicationConfig] = None
        self.injector: Optional[Injector] = None
        self.is_running = False
        self._setup_signal_handlers()

    def start(self, config_path: Optional[str] = None) -> None:
        """
        Start the service with standardized bootstrap sequence.

        Bootstrap sequence:
        1. Load configuration
        2. Setup logging
        3. Initialize dependency injection
        4. Setup health checks
        5. Start service-specific logic
        """
        try:
            logger.info(f"Starting {self.service_name} service...")

            # Step 1: Load configuration
            self._load_configuration(config_path)

            # Step 2: Setup logging
            self._setup_logging()

            # Step 3: Initialize dependency injection
            self._setup_dependency_injection()

            # Step 4: Setup health checks
            self._setup_health_checks()

            # Step 5: Start service-specific logic
            self._start_service()

            self.is_running = True
            logger.info(f"{self.service_name} service started successfully")

            # Keep service running
            self._run_main_loop()

        except Exception as e:
            logger.error(f"Failed to start {self.service_name}: {e}")
            self._cleanup()
            sys.exit(1)

    def stop(self) -> None:
        """Gracefully stop the service."""
        if self.is_running:
            logger.info(f"Stopping {self.service_name} service...")
            self._stop_service()
            self._cleanup()
            self.is_running = False
            logger.info(f"{self.service_name} service stopped")

    def _load_configuration(self, config_path: Optional[str] = None) -> None:
        """Load service configuration using standardized loader."""
        from drl_trading_common.config.service_config_loader import ServiceConfigLoader

        self.config = ServiceConfigLoader.load_config(
            config_class=self.config_class,
            service=self.service_name,
            config_path=config_path
        )
        logger.info(f"Configuration loaded for {self.service_name}")

    def _setup_logging(self) -> None:
        """Setup standardized logging configuration."""
        logging_config = self.config.logging_config if hasattr(self.config, 'logging_config') else None
        StandardLoggingSetup.configure_logging(
            service_name=self.service_name,
            config=logging_config
        )

    def _setup_dependency_injection(self) -> None:
        """Initialize dependency injection container."""
        di_modules = self.get_dependency_modules()
        self.injector = Injector(di_modules)
        logger.info("Dependency injection container initialized")

    def _setup_health_checks(self) -> None:
        """Setup standardized health check endpoints."""
        health_service = self.injector.get(HealthCheckService)
        health_service.register_checks(self.get_health_checks())

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    # Abstract methods for service-specific implementation
    @abstractmethod
    def get_dependency_modules(self) -> List[Module]:
        """Return dependency injection modules for this service."""
        pass

    @abstractmethod
    def _start_service(self) -> None:
        """Start service-specific logic."""
        pass

    @abstractmethod
    def _stop_service(self) -> None:
        """Stop service-specific logic."""
        pass

    @abstractmethod
    def _run_main_loop(self) -> None:
        """Run the main service loop."""
        pass

    def get_health_checks(self) -> List[HealthCheck]:
        """Return health checks for this service (optional override)."""
        return []

    def _cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        pass
```

### 2. Service-Specific Bootstrap Implementations
**Duration:** 1 day

#### Inference Service Bootstrap:
```python
# drl-trading-inference/src/drl_trading_inference/bootstrap/inference_bootstrap.py
from drl_trading_common.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_inference.config.inference_config import InferenceConfig
from drl_trading_inference.di.inference_module import InferenceModule

class InferenceServiceBootstrap(ServiceBootstrap):
    """Bootstrap for DRL Trading Inference Service."""

    def __init__(self):
        super().__init__(
            service_name="inference",
            config_class=InferenceConfig
        )
        self.inference_service = None

    def get_dependency_modules(self) -> List[Module]:
        """Return inference service DI modules."""
        return [
            InferenceModule(self.config),
            CommonModule(self.config),
            MessagingModule(self.config)
        ]

    def _start_service(self) -> None:
        """Start inference service logic."""
        self.inference_service = self.injector.get(InferenceService)
        self.inference_service.start()

    def _stop_service(self) -> None:
        """Stop inference service logic."""
        if self.inference_service:
            self.inference_service.stop()

    def _run_main_loop(self) -> None:
        """Run inference service main loop."""
        # For message-driven services, this might just wait
        import time
        while self.is_running:
            time.sleep(1)

    def get_health_checks(self) -> List[HealthCheck]:
        """Return inference-specific health checks."""
        return [
            ModelHealthCheck(),
            MessageBusHealthCheck(),
            FeatureStoreHealthCheck()
        ]

# Updated main.py
from drl_trading_inference.bootstrap.inference_bootstrap import InferenceServiceBootstrap

def main():
    """Main entry point for inference service."""
    bootstrap = InferenceServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()
```

#### Training Service Bootstrap:
```python
# drl-trading-training/src/drl_trading_training/bootstrap/training_bootstrap.py
from drl_trading_common.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_training.config.training_config import TrainingConfig

class TrainingServiceBootstrap(ServiceBootstrap):
    """Bootstrap for DRL Trading Training Service."""

    def __init__(self):
        super().__init__(
            service_name="training",
            config_class=TrainingConfig
        )
        self.training_app = None

    def get_dependency_modules(self) -> List[Module]:
        """Return training service DI modules."""
        return [
            TrainingModule(self.config),
            CoreModule(self.config),
            StrategyModule(self.config)
        ]

    def _start_service(self) -> None:
        """Start training service logic."""
        self.training_app = self.injector.get(TrainingApp)
        # Training is typically a one-time job
        self.training_app.execute_training()

    def _stop_service(self) -> None:
        """Stop training service logic."""
        if self.training_app:
            self.training_app.stop()

    def _run_main_loop(self) -> None:
        """Training services typically exit after completion."""
        # For training, we might just wait for completion
        if self.training_app:
            self.training_app.wait_for_completion()
        self.stop()  # Auto-stop after training completion

# Updated main.py
from drl_trading_training.bootstrap.training_bootstrap import TrainingServiceBootstrap

def main():
    """Main entry point for training service."""
    bootstrap = TrainingServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()
```

### 3. Standardized Logging Setup
**Duration:** 0.5 days

```python
# drl_trading_common/logging/standard_logging_setup.py
import logging
import logging.config
import os
from typing import Optional, Dict, Any

class StandardLoggingSetup:
    """Standardized logging configuration for all services."""

    DEFAULT_LOG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    @classmethod
    def configure_logging(
        cls,
        service_name: str,
        config: Optional[Dict[str, Any]] = None,
        log_level: Optional[str] = None
    ) -> None:
        """Configure standardized logging for a service."""

        # Determine log level
        log_level = log_level or os.environ.get("LOG_LEVEL", "INFO")

        # Default logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": cls.DEFAULT_LOG_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "service": "' + service_name + '", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
                    "datefmt": "%Y-%m-%dT%H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": log_level,
                    "formatter": "json",
                    "filename": f"logs/{service_name}.log",
                    "mode": "a"
                }
            },
            "loggers": {
                service_name: {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_common": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "drl_trading_core": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"]
            }
        }

        # Override with custom config if provided
        if config:
            logging_config.update(config)

        # Apply configuration
        logging.config.dictConfig(logging_config)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        logger = logging.getLogger(service_name)
        logger.info(f"Logging configured for {service_name} service")
```

### 4. Health Check Framework
**Duration:** 0.5 days

```python
# drl_trading_common/health/health_check.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class HealthCheck(ABC):
    """Abstract base for health checks."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dict with 'status', 'message', and optional details
        """
        pass

class HealthCheckService:
    """Service for managing and executing health checks."""

    def __init__(self):
        self.checks: List[HealthCheck] = []

    def register_checks(self, checks: List[HealthCheck]) -> None:
        """Register health checks."""
        self.checks.extend(checks)

    def check_health(self) -> Dict[str, Any]:
        """Execute all health checks and return aggregated status."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for check in self.checks:
            try:
                result = check.check()
                results[check.name] = result

                if result.get("status") == HealthStatus.UNHEALTHY.value:
                    overall_status = HealthStatus.UNHEALTHY
                elif (result.get("status") == HealthStatus.DEGRADED.value and
                      overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                results[check.name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Health check failed: {str(e)}"
                }
                overall_status = HealthStatus.UNHEALTHY

        return {
            "status": overall_status.value,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }

# Example health checks
class DatabaseHealthCheck(HealthCheck):
    def __init__(self, db_service):
        super().__init__("database")
        self.db_service = db_service

    def check(self) -> Dict[str, Any]:
        try:
            # Test database connection
            self.db_service.ping()
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Database connection failed: {str(e)}"
            }
```

## Migration Plan

### Phase 1: Framework Implementation (1 day)
1. Implement `ServiceBootstrap` base class
2. Implement `StandardLoggingSetup`
3. Implement `HealthCheckService` framework
4. Create unit tests for framework components

### Phase 2: Service Migration (2 days)
1. **Migrate drl-trading-inference** (simplest service)
   - Create `InferenceServiceBootstrap`
   - Update `main.py`
   - Add health checks
   - Test thoroughly

2. **Migrate drl-trading-training**
   - Create `TrainingServiceBootstrap`
   - Update `main.py`
   - Handle training-specific patterns

3. **Migrate remaining services**
   - drl-trading-ingest
   - drl-trading-execution
   - drl-trading-preprocess

### Phase 3: Validation & Documentation
1. Integration testing across all services
2. Performance validation
3. Documentation updates
4. Developer guidelines

## Testing Strategy

### Unit Tests
```python
class TestServiceBootstrap:
    def test_configuration_loading(self):
        """Test configuration loading in bootstrap sequence."""
        # Given
        bootstrap = TestServiceBootstrap()

        # When
        bootstrap._load_configuration("test_config.yaml")

        # Then
        assert bootstrap.config is not None
        assert isinstance(bootstrap.config, TestConfig)

    def test_dependency_injection_setup(self):
        """Test DI container initialization."""
        # Given
        bootstrap = TestServiceBootstrap()
        bootstrap.config = TestConfig()

        # When
        bootstrap._setup_dependency_injection()

        # Then
        assert bootstrap.injector is not None
        test_service = bootstrap.injector.get(TestService)
        assert test_service is not None

    def test_graceful_shutdown(self):
        """Test graceful shutdown handling."""
        # Given
        bootstrap = TestServiceBootstrap()
        bootstrap.start()

        # When
        bootstrap.stop()

        # Then
        assert not bootstrap.is_running
```

## Acceptance Criteria
- [ ] All services use identical bootstrap patterns with `ServiceBootstrap` base class
- [ ] Standardized configuration loading integrated with secret management
- [ ] Consistent logging configuration across all services
- [ ] Graceful shutdown handling with SIGTERM/SIGINT support
- [ ] Health check framework implemented with service-specific checks
- [ ] All existing services migrated without functionality loss
- [ ] Performance impact is minimal (< 100ms additional startup time)
- [ ] Comprehensive test coverage for bootstrap framework
- [ ] Documentation and migration guides complete

## Dependencies
- **Depends On:** T002 (Config loader), T003 (Secret management)
- **Blocks:** T005 (Logging standardization), T006 (Service migration)

## Risks
- **Service Disruption**: Migration could break existing services
  - **Mitigation**: Incremental migration with thorough testing
- **Performance Impact**: Additional bootstrap overhead
  - **Mitigation**: Performance benchmarking and optimization
- **Complexity Overhead**: Too much abstraction
  - **Mitigation**: Keep framework simple and focused

## Definition of Done
- [ ] `ServiceBootstrap` framework implemented and tested
- [ ] All services successfully migrated to new bootstrap pattern
- [ ] Health check endpoints working for all services
- [ ] Graceful shutdown tested for all services
- [ ] No functionality regression in any migrated service
- [ ] Performance benchmarks show acceptable impact
- [ ] Code review and security review completed
- [ ] Documentation complete with migration examples
