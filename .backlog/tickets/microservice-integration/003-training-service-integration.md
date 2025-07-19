# Training Service Integration with Feature Pipeline

**Epic:** Microservice Integration Pipeline
**Status:** 📝 Todo
**Assignee:** You
**Estimated:** 10 hours

## Description
Integrate drl-trading-training service with feature pipeline and microservice ecosystem. Support two training modes: (1) from-scratch training with full data/feature delegation, and (2) continue-training with existing model/features from MLflow/Feast. Focus on model training, checkpointing, and backtesting evaluation.

## Acceptance Criteria
**From-Scratch Training Mode:**
- [ ] Request data/features from ingest service (delegate completely)
- [ ] Wait for preprocessing completion notification
- [ ] Start training when features ready
- [ ] Model checkpointing during training

**Continue-Training Mode:**
- [ ] Fetch existing model from MLflow model store
- [ ] Fetch corresponding features from Feast using model version config
- [ ] Resume training from checkpoint
- [ ] Model versioning and checkpointing

**Common Training Pipeline:**
- [ ] Integration with existing CoreEngine and AgentTrainingService
- [ ] Model evaluation using drl-trading-core backtesting module
- [ ] Training metrics logging and tracking
- [ ] Graceful error handling for both modes

## Technical Notes
**From-Scratch Mode:**
- Synchronous API call to ingest: "I need data for FeatureConfig X"
- Ingest handles data acquisition and triggers preprocessing
- Async notification from preprocessing: "Features ready for training"
- Training starts immediately upon notification

**Continue-Training Mode:**
- Query MLflow for existing model and metadata
- Use model's FeatureConfigVersionInfo to fetch from Feast
- Resume training with existing model weights
- Maintain version lineage for model evolution

**Core Integration:**
- Model checkpointing with MLflow integration
- Backtesting evaluation using drl-trading-core modules
- Training orchestration without data/feature management
- Clean separation of concerns

## Files to Change
- [ ] `drl-trading-training/src/service/training_orchestration_service.py`
- [ ] `drl-trading-training/src/service/model_lifecycle_service.py`
- [ ] `drl-trading-training/src/service/backtesting_evaluation_service.py`
- [ ] `drl-trading-training/src/client/ingest_client.py`
- [ ] `drl-trading-training/src/client/feast_client.py`
- [ ] `drl-trading-training/src/client/mlflow_client.py`
- [ ] `drl-trading-training/src/messaging/preprocessing_listener.py`
- [ ] `tests/integration/training/from_scratch_training_test.py`
- [ ] `tests/integration/training/continue_training_test.py`

## Dependencies
- Feature Pipeline Infrastructure epic (offline Feast store)
- **MLflow Model Management Integration epic** (model store and versioning)
- drl-trading-ingest API endpoints for data requests
- drl-trading-preprocess messaging for feature notifications
- drl-trading-core backtesting modules
- Existing CoreEngine and AgentTrainingService

## Definition of Done
- [ ] From-scratch training mode working end-to-end
- [ ] Continue-training mode with MLflow integration
- [ ] Feature fetching from Feast by model version
- [ ] Model checkpointing and versioning functional
- [ ] Backtesting evaluation integrated
- [ ] Training metrics logged properly
- [ ] Both training modes tested
- [ ] Tests pass (mypy + ruff)
- [ ] Clean service communication protocols
