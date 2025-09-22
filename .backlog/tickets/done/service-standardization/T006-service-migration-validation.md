# T006: Service Migration & Validation

**Priority:** High
**Estimated Effort:** 4 days
**Type:** Migration & Testing
**Depends On:** T002, T003, T004, T005

## Objective
Migrate all existing DRL trading microservices to use the standardized configuration, bootstrap, and logging patterns, followed by comprehensive validation to ensure no functionality regression.

## Scope
Systematic migration of all services in the DRL trading ecosystem to adopt:
- Unified configuration management (T002)
- Standardized sensitive data handling (T003)
- Common bootstrap patterns (T004)
- Consistent logging (T005)

## Service Migration Plan

### Phase 1: Framework Preparation & Testing
**Duration:** 1 day

#### Prerequisites Validation:
```bash
# Validate all dependencies are ready
./scripts/validate-service-standards.sh

# Expected frameworks to be available:
- drl_trading_common.config.service_config_loader
- drl_trading_common.config.secret_manager
- drl_trading_common.bootstrap.service_bootstrap
- drl_trading_common.logging.service_logger
```

#### Migration Testing Framework:
```python
# tests/migration/service_migration_test_framework.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import subprocess
import time
import requests
import logging

class ServiceMigrationTest(ABC):
    """Base class for service migration testing."""

    def __init__(self, service_name: str, service_path: str):
        self.service_name = service_name
        self.service_path = service_path
        self.logger = logging.getLogger(f"migration.{service_name}")

    @abstractmethod
    def get_pre_migration_health(self) -> Dict[str, Any]:
        """Get service health before migration."""
        pass

    @abstractmethod
    def validate_post_migration_functionality(self) -> bool:
        """Validate service functionality after migration."""
        pass

    def run_migration_test(self) -> bool:
        """Execute complete migration test cycle."""
        try:
            # 1. Capture pre-migration state
            self.logger.info(f"Capturing pre-migration state for {self.service_name}")
            pre_state = self.get_pre_migration_health()

            # 2. Backup current service
            self._backup_service()

            # 3. Apply migration
            self.logger.info(f"Applying migration to {self.service_name}")
            self._apply_migration()

            # 4. Validate post-migration functionality
            self.logger.info(f"Validating post-migration functionality for {self.service_name}")
            if not self.validate_post_migration_functionality():
                self.logger.error(f"Post-migration validation failed for {self.service_name}")
                self._rollback_migration()
                return False

            # 5. Performance comparison
            if not self._validate_performance():
                self.logger.warning(f"Performance regression detected in {self.service_name}")

            self.logger.info(f"Migration successful for {self.service_name}")
            return True

        except Exception as e:
            self.logger.error(f"Migration failed for {self.service_name}: {e}")
            self._rollback_migration()
            return False

    def _backup_service(self):
        """Backup current service implementation."""
        backup_path = f"{self.service_path}.backup"
        subprocess.run(["cp", "-r", self.service_path, backup_path], check=True)

    def _rollback_migration(self):
        """Rollback to backup if migration fails."""
        backup_path = f"{self.service_path}.backup"
        subprocess.run(["rm", "-rf", self.service_path], check=True)
        subprocess.run(["mv", backup_path, self.service_path], check=True)

    def _apply_migration(self):
        """Apply the migration (implemented by subclasses)."""
        pass

    def _validate_performance(self) -> bool:
        """Validate performance hasn't regressed significantly."""
        # Basic startup time validation
        start_time = time.time()
        self._start_service()
        startup_time = time.time() - start_time

        # Allow up to 500ms additional startup time
        return startup_time < 5.0  # 5 seconds max startup
```

### Phase 2: Individual Service Migrations
**Duration:** 2.5 days

#### 2.1 drl-trading-inference Migration (0.5 days)
**Complexity:** Low - Cleanest existing patterns

##### Current State Analysis:
```python
# Current drl-trading-inference/main.py
from drl_trading_inference import setup_logging, bootstrap_inference_service

if __name__ == "__main__":
    setup_logging()
    bootstrap_inference_service()
```

##### Migration Implementation:
```python
# New drl-trading-inference/main.py
from drl_trading_inference.bootstrap.inference_bootstrap import InferenceServiceBootstrap

def main():
    """Main entry point for inference service."""
    bootstrap = InferenceServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()

# drl-trading-inference/src/drl_trading_inference/bootstrap/inference_bootstrap.py
from drl_trading_common.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_inference.config.inference_config import InferenceConfig
from drl_trading_inference.di.inference_module import InferenceModule
from drl_trading_common.di.common_module import CommonModule

class InferenceServiceBootstrap(ServiceBootstrap):
    """Bootstrap for DRL Trading Inference Service."""

    def __init__(self):
        super().__init__(
            service_name="inference",
            config_class=InferenceConfig
        )
        self.inference_service = None

    def get_dependency_modules(self) -> List[Module]:
        return [
            InferenceModule(self.config),
            CommonModule(self.config)
        ]

    def _start_service(self) -> None:
        self.inference_service = self.injector.get(InferenceService)
        self.inference_service.start()

    def _stop_service(self) -> None:
        if self.inference_service:
            self.inference_service.stop()

    def _run_main_loop(self) -> None:
        import time
        while self.is_running:
            time.sleep(1)

# Update inference configuration
# drl-trading-inference/src/drl_trading_inference/config/inference_config.py
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.logging_config import LoggingConfig
from pydantic import Field

class InferenceConfig(BaseApplicationConfig):
    """Configuration for inference service."""

    # Service-specific configuration
    model_path: str = Field(..., description="Path to trained model")
    batch_size: int = Field(default=32, description="Inference batch size")
    max_latency_ms: int = Field(default=100, description="Maximum inference latency")

    # Standard configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

##### Migration Testing:
```python
class InferenceMigrationTest(ServiceMigrationTest):
    """Migration test for inference service."""

    def __init__(self):
        super().__init__("inference", "drl-trading-inference")

    def get_pre_migration_health(self) -> Dict[str, Any]:
        """Get inference service health before migration."""
        # Test inference endpoint if available
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def validate_post_migration_functionality(self) -> bool:
        """Validate inference service functionality after migration."""
        try:
            # Start service
            self._start_service()

            # Wait for startup
            time.sleep(3)

            # Test health endpoint
            response = requests.get("http://localhost:8080/health", timeout=10)
            if response.status_code != 200:
                return False

            # Test inference endpoint if available
            # (Add specific inference validation here)

            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
        finally:
            self._stop_service()
```

#### 2.2 drl-trading-training Migration (0.5 days)
**Complexity:** Medium - CLI-focused service

##### Current State Analysis:
```python
# Current drl-trading-training/main.py
from .src.drl_trading_training import TrainingApp

if __name__ == "__main__":
    app = TrainingApp()
    app.run()
```

##### Migration Implementation:
```python
# New drl-trading-training/main.py
from drl_trading_training.bootstrap.training_bootstrap import TrainingServiceBootstrap

def main():
    """Main entry point for training service."""
    bootstrap = TrainingServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()

# drl-trading-training/src/drl_trading_training/bootstrap/training_bootstrap.py
class TrainingServiceBootstrap(ServiceBootstrap):
    """Bootstrap for DRL Trading Training Service."""

    def __init__(self):
        super().__init__(
            service_name="training",
            config_class=TrainingConfig
        )
        self.training_app = None

    def get_dependency_modules(self) -> List[Module]:
        return [
            TrainingModule(self.config),
            CoreModule(self.config),
            StrategyModule(self.config)
        ]

    def _start_service(self) -> None:
        self.training_app = self.injector.get(TrainingApp)
        # Training is typically a one-time job
        self.training_app.start_training()

    def _stop_service(self) -> None:
        if self.training_app:
            self.training_app.stop_training()

    def _run_main_loop(self) -> None:
        """Training services typically exit after completion."""
        if self.training_app:
            self.training_app.wait_for_completion()
        self.stop()  # Auto-stop after training completion
```

#### 2.3 drl-trading-ingest Migration (0.75 days)
**Complexity:** High - Flask web service with database

##### Current State Analysis:
```python
# Current drl-trading-ingest/main.py
import logging
from flask import Flask
from injector import Injector
from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule

def create_app():
    # Flask-specific bootstrap logic
```

##### Migration Implementation:
```python
# New drl-trading-ingest/main.py
from drl_trading_ingest.bootstrap.ingest_bootstrap import IngestServiceBootstrap

def main():
    """Main entry point for ingest service."""
    bootstrap = IngestServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()

# drl-trading-ingest/src/drl_trading_ingest/bootstrap/ingest_bootstrap.py
from flask import Flask
from drl_trading_common.bootstrap.service_bootstrap import ServiceBootstrap

class IngestServiceBootstrap(ServiceBootstrap):
    """Bootstrap for DRL Trading Ingest Service."""

    def __init__(self):
        super().__init__(
            service_name="ingest",
            config_class=IngestConfig
        )
        self.flask_app = None
        self.server = None

    def get_dependency_modules(self) -> List[Module]:
        return [
            IngestModule(self.config),
            CommonModule(self.config),
            DatabaseModule(self.config)
        ]

    def _start_service(self) -> None:
        # Create Flask app with DI
        self.flask_app = self._create_flask_app()

        # Start Flask server
        host = self.config.server.host
        port = self.config.server.port
        self.server = self.flask_app.run(host=host, port=port, threaded=True)

    def _stop_service(self) -> None:
        if self.server:
            self.server.shutdown()

    def _run_main_loop(self) -> None:
        """Flask handles the main loop."""
        if self.flask_app:
            # Flask app runs in its own thread
            import time
            while self.is_running:
                time.sleep(1)

    def _create_flask_app(self) -> Flask:
        """Create Flask app with dependency injection."""
        app = Flask(__name__)

        # Configure Flask with injector
        app.injector = self.injector

        # Register routes
        from drl_trading_ingest.adapter.web.routes import register_routes
        register_routes(app)

        return app
```

#### 2.4 Remaining Services Migration (0.75 days)
- **drl-trading-execution**: Similar to inference service
- **drl-trading-preprocess**: CLI service similar to training

### Phase 3: Integration Testing & Validation
**Duration:** 0.5 days

#### Cross-Service Integration Tests:
```python
# tests/migration/cross_service_integration_test.py
class CrossServiceIntegrationTest:
    """Test service interactions after migration."""

    def test_service_communication(self):
        """Test that services can still communicate after migration."""
        # Given
        services = ['inference', 'ingest', 'execution']

        for service in services:
            # When
            self._start_service(service)

        # Then
        # Test message bus communication
        # Test health check endpoints
        # Test service dependencies

        for service in services:
            self._stop_service(service)

    def test_configuration_consistency(self):
        """Test that all services use consistent configuration patterns."""
        # Validate all services can load their configurations
        # Validate secret substitution works
        # Validate environment-specific configurations

    def test_logging_correlation(self):
        """Test that log correlation works across services."""
        # Start services with correlation context
        # Generate cross-service activity
        # Validate correlation IDs appear in all service logs
```

## Rollback Strategy

### Automated Rollback:
```python
# scripts/rollback_service_migration.py
import argparse
import subprocess
import logging

def rollback_service(service_name: str) -> bool:
    """Rollback a service to pre-migration state."""
    try:
        service_path = f"drl-trading-{service_name}"
        backup_path = f"{service_path}.backup"

        if not os.path.exists(backup_path):
            logging.error(f"No backup found for {service_name}")
            return False

        # Stop current service
        subprocess.run(["docker-compose", "stop", service_name], check=True)

        # Restore backup
        subprocess.run(["rm", "-rf", service_path], check=True)
        subprocess.run(["mv", backup_path, service_path], check=True)

        # Restart service
        subprocess.run(["docker-compose", "start", service_name], check=True)

        logging.info(f"Successfully rolled back {service_name}")
        return True

    except Exception as e:
        logging.error(f"Rollback failed for {service_name}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("service", help="Service name to rollback")
    args = parser.parse_args()

    rollback_service(args.service)
```

## Performance Validation

### Startup Time Comparison:
```python
# tests/migration/performance_validation.py
class PerformanceValidationTest:
    """Validate performance hasn't regressed after migration."""

    def test_startup_time_regression(self):
        """Test that service startup time hasn't significantly increased."""
        # Measure startup times for each service
        # Compare with baseline (pre-migration)
        # Fail if regression > 20%

    def test_memory_usage_regression(self):
        """Test that memory usage hasn't significantly increased."""
        # Monitor memory usage during startup and runtime
        # Compare with baseline
        # Fail if regression > 15%

    def test_functionality_performance(self):
        """Test that core functionality performance is maintained."""
        # Test inference latency
        # Test training throughput
        # Test data ingestion rates
```

## Documentation Updates

### Service Migration Guide:
```markdown
# Service Migration Guide

## Overview
This guide documents the migration of all DRL trading services to standardized patterns.

## Migration Checklist per Service:
- [ ] Configuration updated to use ServiceConfigLoader
- [ ] Secret management implemented
- [ ] Bootstrap pattern applied
- [ ] Logging standardized
- [ ] Health checks implemented
- [ ] Integration tests passing
- [ ] Performance validated

## Service-Specific Notes:
### drl-trading-inference
- Configuration path: `config/inference.{env}.yaml`
- Health endpoint: `/health`
- Key dependencies: Model files, Feature store

### drl-trading-training
- Configuration path: `config/training.{env}.yaml`
- CLI execution: `python -m drl_trading_training`
- Key dependencies: Training data, Strategy modules

...
```

## Acceptance Criteria
- [ ] All services successfully migrated to standardized patterns
- [ ] No functionality regression in any service
- [ ] All service configurations use unified pattern
- [ ] Secret management working across all services
- [ ] Logging is consistent and structured across all services
- [ ] Health checks implemented for all applicable services
- [ ] Cross-service integration still functional
- [ ] Performance regression < 20% for startup time, < 15% for memory
- [ ] Rollback capability validated for all services
- [ ] Migration documentation complete
- [ ] Service-specific configuration examples provided

## Dependencies
- **Depends On:** T002, T003, T004, T005 (All framework components)
- **Blocks:** T007 (Documentation - needs migrated examples)

## Risks
- **Service Downtime**: Migration could cause service interruptions
  - **Mitigation**: Rolling migration with immediate rollback capability
- **Integration Breakage**: Service interactions could break
  - **Mitigation**: Comprehensive integration testing
- **Performance Regression**: New patterns could slow services
  - **Mitigation**: Performance benchmarking and optimization
- **Configuration Errors**: Migration could introduce configuration issues
  - **Mitigation**: Extensive configuration validation and testing

## Definition of Done
- [ ] All services migrated and validated
- [ ] No functionality regression in any service
- [ ] Cross-service integration tests passing
- [ ] Performance benchmarks within acceptable ranges
- [ ] Health checks functional for all services
- [ ] Secret management working in all environments
- [ ] Logging consistency achieved across all services
- [ ] Rollback procedures tested and documented
- [ ] Migration guide complete with troubleshooting
- [ ] Production deployment plan ready
