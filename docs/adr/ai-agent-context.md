# DRL Trading Architecture Decisions Summary

This file provides AI agents with quick context on key architectural decisions in the DRL Trading system.

## Overview
The DRL Trading system uses Architecture Decision Records (ADRs) to document significant architectural choices. These decisions provide context for development recommendations and ensure consistency across the microservice architecture.

## Key Architectural Patterns

### Documentation & Process
- **ADR-0001: ADR System Adoption** (Accepted)
  - Establishes systematic documentation of architectural decisions
  - Provides AI agent context for better assistance
  - Uses markdown-based ADRs in version control

### Configuration & Infrastructure
- **ADR-0002: Configuration Architecture Standardization** (Accepted)
  - Standardizes on YAML configuration format across all services
  - Enhances existing EnhancedServiceConfigLoader with environment support
  - Maintains Pydantic schema validation patterns
  - Adds secret substitution using ${VAR:default} syntax

### Architecture Patterns
- **Microservice Architecture**: Independent services for different trading concerns
  - drl-trading-core: Framework and common functionality
  - drl-trading-common: Shared messaging and data models
  - drl-trading-ingest: Market data ingestion
  - drl-trading-inference: Real-time prediction service
  - drl-trading-training: Model training service
  - drl-trading-execution: Trade execution service
  - drl-trading-strategy-*: Pluggable strategy modules

### Current Decisions in Progress
- **Service Standardization Epic**: Establishing unified patterns across services
  - Configuration management standardization
  - Service bootstrap patterns
  - Logging and observability standards
  - Secret management approaches

### Technology Choices
- **Dependency Injection**: Python Injector library for testable, modular code
- **Feature Store**: Feast for ML feature consistency and versioning
- **Messaging**: Event-driven architecture with pluggable transport layer

## AI Assistant Guidelines
When providing architectural recommendations:

1. **Check Relevant ADRs**: Reference established decisions before suggesting changes
2. **Maintain Consistency**: Recommend patterns that align with documented decisions
3. **Suggest ADR Creation**: For significant architectural discussions, recommend creating new ADRs
4. **Reference Numbers**: Use ADR numbers when explaining architectural choices
5. **Consider Current Epic**: Be aware of ongoing service standardization work

## Status Legend
- **Accepted**: Decision is approved and should be followed
- **Proposed**: Decision is under consideration
- **Deprecated**: Decision is no longer recommended
- **Superseded**: Decision has been replaced by a newer ADR

## Quick Links
- [Full ADR Index](README.md)
- [ADR Template](template.md)
- [Service Standardization Epic](../.backlog/tickets/service-standardization/)

---
*This summary is automatically updated when ADRs are added or modified.*
