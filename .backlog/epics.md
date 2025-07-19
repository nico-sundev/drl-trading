# Epics
Priority top down

## Feature Pipeline Infrastructure
**Status:** 🔄 In Progress
**Priority:** High
**Description:** ML pipeline backbone for feature fetching/loading across preprocessing, training, and inference. Enables unified feature access for entire DRL trading system.

**Tickets:** See `tickets/feast-implementation/` for detailed implementation progress

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
