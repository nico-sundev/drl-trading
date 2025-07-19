# Model Versioning and FeatureConfigVersionInfo Integration

**Epic:** MLflow Model Management Integration
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 6 hours

## Description
Implement model versioning system that links MLflow models with FeatureConfigVersionInfo for complete traceability. Enable model-to-feature configuration mapping and version lineage tracking across training iterations.

## Acceptance Criteria
- [ ] Model metadata includes FeatureConfigVersionInfo
- [ ] Model versioning with semantic versioning support
- [ ] Model lineage tracking (parent model references)
- [ ] Feature configuration retrieval by model version
- [ ] Model tagging with feature config hash
- [ ] Version compatibility checking utilities
- [ ] Model artifact organization and naming conventions
- [ ] Query interface for model-feature relationships

## Technical Notes
- Store FeatureConfigVersionInfo as model metadata in MLflow
- Use consistent naming conventions for model versions
- Implement model lineage for continue-training scenarios
- Create utility functions for model-feature lookups
- Ensure feature config compatibility validation
- Support both fresh and continue-training model creation

## Files to Change
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/service/model_versioning_service.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/service/model_metadata_service.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/interface/model_versioning_interface.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/util/version_utils.py`
- [ ] `drl-trading-core/src/drl_trading_core/mlflow/util/compatibility_checker.py`
- [ ] `drl-trading-common/src/drl_trading_common/model/model_version_info.py`
- [ ] `tests/unit/mlflow/service/model_versioning_service_test.py`
- [ ] `tests/unit/mlflow/util/version_utils_test.py`
- [ ] `tests/integration/mlflow/model_feature_integration_test.py`

## Dependencies
- Core MLflow Integration (002)
- FeatureConfigVersionInfo model (existing)
- Model Registry Service implementation

## Definition of Done
- [ ] Model-feature config linking working
- [ ] Model versioning system functional
- [ ] Lineage tracking implemented
- [ ] Compatibility checking working
- [ ] Metadata storage and retrieval functional
- [ ] Version utilities comprehensive
- [ ] Tests pass (mypy + ruff)
- [ ] Integration tests covering model-feature scenarios
- [ ] Documentation for versioning conventions
