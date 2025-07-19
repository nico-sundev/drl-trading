# MLflow Model Management Integration Epic

**Status:** 📝 Planned
**Priority:** High
**Description:** Complete MLflow integration for ML lifecycle management with GitLab hosting

## Overview
Establishes centralized model management with GitLab-hosted MLflow, enabling model versioning, experiment tracking, and artifact storage. Integrates with FeatureConfigVersionInfo for complete model-config traceability across the DRL trading system.

## Progress Tracking
- [ ] GitLab MLflow Setup and Configuration
- [ ] Core MLflow Integration in drl-trading-core
- [ ] Model Store and Versioning System
- [ ] Experiment Tracking with Backtesting Integration
- [ ] FeatureConfigVersionInfo Integration
- [ ] Training Service MLflow Integration
- [ ] Inference Service MLflow Integration

## Tickets
- [001-gitlab-mlflow-setup.md](./001-gitlab-mlflow-setup.md) - 📝 Todo
- [002-core-mlflow-integration.md](./002-core-mlflow-integration.md) - 📝 Todo
- [003-model-versioning-system.md](./003-model-versioning-system.md) - 📝 Todo
- [004-experiment-tracking.md](./004-experiment-tracking.md) - 📝 Todo
- [005-backtesting-integration.md](./005-backtesting-integration.md) - 📝 Todo

## Dependencies
- GitLab account and MLflow hosting setup
- FeatureConfigVersionInfo model (existing)
- drl-trading-core backtesting module
- Training and inference service integration points

## Strategic Value
- **Model Lifecycle Management**: Complete model versioning and tracking
- **Experiment Reproducibility**: Consistent experiment storage and retrieval
- **Service Integration**: Unified model access across training/inference
- **Configuration Traceability**: Links models to exact feature configurations
