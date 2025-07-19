# Core MLflow Integration in drl-trading-core

**Epic:** MLflow Model Management Integration
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 8 hours

## Description
Implement core MLflow integration within drl-trading-core framework, providing centralized model management services that can be used by training, inference, and other services. Focus on dependency injection patterns and service abstractions.

## Acceptance Criteria
- [ ] MLflow client abstraction layer implemented
- [ ] Dependency injection configuration for MLflow services
- [ ] Model registry operations (save, load, version, tag)
- [ ] Experiment tracking operations (log metrics, parameters, artifacts)
- [ ] Configuration management for MLflow connection
- [ ] Error handling and retry logic for MLflow operations
- [ ] Integration with existing CoreModule DI container
- [ ] Service interfaces for clean abstraction

## Technical Notes
- Follow existing DI patterns in drl-trading-core
- Create service interfaces for testability
- Implement proper error handling and logging
- Use configuration from ApplicationConfig pattern
- Ensure thread-safety for concurrent operations
- Abstract MLflow operations behind interfaces

## Files to Change
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/client/mlflow_client.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/service/model_registry_service.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/service/experiment_tracking_service.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/interface/mlflow_client_interface.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/interface/model_registry_interface.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/interface/experiment_tracking_interface.py`
- [ ] `drl-trading-core/src/drl_trading_core/common/di/core_module.py`
- [ ] `drl-trading-common/src/drl_trading_common/config/mlflow_config.py`
- [ ] `tests/unit/mlflow/client/mlflow_client_test.py`
- [ ] `tests/unit/mlflow/service/model_registry_service_test.py`
- [ ] `tests/integration/mlflow/mlflow_integration_test.py`

## Dependencies
- GitLab MLflow Setup and Configuration (001)
- MLflow Python client library
- Existing drl-trading-core DI infrastructure
- ApplicationConfig pattern from drl-trading-common

## Definition of Done
- [ ] MLflow client services implemented and testable
- [ ] Dependency injection working correctly
- [ ] Model registry operations functional
- [ ] Experiment tracking operations functional
- [ ] Configuration loading working
- [ ] Error handling comprehensive
- [ ] Tests pass (mypy + ruff)
- [ ] Integration tests with MLflow server passing
- [ ] Service abstractions clean and reusable
