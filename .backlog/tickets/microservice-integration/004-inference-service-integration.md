# Inference Service Integration with Online Features

**Epic:** Microservice Integration Pipeline
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 10 hours

## Description
Integrate drl-trading-inference service with feature pipeline for online feature fetching and materialization. Implement model loading, real-time feature access, and prediction pipeline to execution service.

## Acceptance Criteria
- [ ] Inference service loads DRL models at startup
- [ ] Online feature fetching via Feast implemented
- [ ] Feature materialization from preprocessing integrated
- [ ] Real-time feature access for inference
- [ ] Prediction pipeline to execution service
- [ ] Async Kafka messaging for predictions
- [ ] Model inference with latest features

## Technical Notes
- Online feature store integration required
- Materialization coordination with preprocessing
- Model loading and management
- Real-time feature access patterns
- Kafka async messaging to execution

## Files to Change
- [ ] `drl-trading-inference/src/service/inference_service.py`
- [ ] `drl-trading-inference/src/service/model_manager.py`
- [ ] `drl-trading-inference/src/service/feature_client.py`
- [ ] `drl-trading-inference/src/client/execution_client.py`
- [ ] `tests/integration/inference/feature_integration_test.py`

## Dependencies
- Feature Pipeline Infrastructure epic (online store)
- drl-trading-preprocess materialization
- drl-trading-execution API contracts
- Kafka messaging infrastructure

## Definition of Done
- [ ] Models load correctly at startup
- [ ] Online features accessible
- [ ] Real-time inference working
- [ ] Predictions sent to execution
- [ ] Tests pass (mypy + ruff)
- [ ] E2E inference pipeline functional
