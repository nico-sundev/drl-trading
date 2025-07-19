# Epics
Priority top down

## Feature Pipeline Infrastructure
**Status:** 🔄 In Progress (95% Complete)
**Priority:** High
**Description:** ML pipeline backbone for feature fetching/loading across preprocessing, training, and inference. Enables unified feature access for entire DRL trading system.

**Completed:** Feast integration with save/fetch repositories, local/S3 backends, online/offline storage, comprehensive unit testing.

**Remaining:** Integration tests need debugging - fixture setup and conftest dependencies require fixes.

**Tickets:** See `tickets/feast-implementation/` - Core implementation complete, integration tests in progress

## MLflow Model Management Integration
**Status:** 📝 Planned
**Priority:** High
**Description:** Centralized ML lifecycle management with GitLab-hosted MLflow. Enables model versioning, experiment tracking, and artifact storage integrated with FeatureConfigVersionInfo for complete model-config traceability.

**Tickets:** See `tickets/mlflow-integration/` for implementation details

## Microservice Integration Pipeline
**Status:** 📝 Planned
**Priority:** High
**Description:** End-to-end microservice integration connecting data ingestion → preprocessing → training/inference → execution via Kafka messaging. Implements complete ML pipeline workflow.

**Tickets:** See `tickets/microservice-integration/` for implementation details

## DRL Trading Architecture Documentation
**Status:** 📝 Planned
**Priority:** Medium
**Description:** Comprehensive architecture documentation including E2E ML pipeline flow diagrams and service dependency maps. Critical for team understanding and AI agent context.

**Tickets:** See `tickets/architecture-documentation/` for documentation tasks

## ML Pipeline Enhancement
**Status:** 📝 Planned
**Priority:** High
**Description:** Enhance DRL training pipeline with feature normalization, RNN/LSTM encoding, and training optimization. Improves model performance and training efficiency.

**Focus Areas:** Feature normalization (ATR), RNN feature embeddings, vectorized environment optimization.

**Tickets:** See `tickets/ml-pipeline-enhancement/` for implementation details

## Data Infrastructure Expansion
**Status:** 📝 Planned
**Priority:** High
**Description:** Expand data ingestion with multiple providers (Binance, TwelveData) and implement robust data pipeline infrastructure for comprehensive market coverage.

**Focus Areas:** Multi-provider APIs, data catchup mechanisms, unified data abstraction.

**Tickets:** See `tickets/data-infrastructure-expansion/` for implementation details

## Observability & Monitoring
**Status:** 📝 Planned
**Priority:** Medium
**Description:** Implement comprehensive observability stack with OpenTelemetry, Jaeger, and Grafana for production monitoring and performance optimization.

**Focus Areas:** Distributed tracing, metrics collection, performance dashboards, alerting.

**Tickets:** See `tickets/observability-monitoring/` for implementation details

## Advanced ML Research
**Status:** 📝 Planned
**Priority:** Future (Research)
**Description:** Explore cutting-edge ML techniques including Graph Neural Networks for pattern recognition and SBX optimization for enhanced trading performance.

**Focus Areas:** GNN pattern recognition, SBX training optimization, advanced feature embeddings.

**Tickets:** See `tickets/advanced-ml-research/` for research initiatives
