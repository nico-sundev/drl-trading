---
applyTo: '**'
---
# Architecture Instructions for AI Agents

This file provides essential architectural guidance to help AI agents understand strategic architectural decisions and maintain consistency when working with this DRL Trading project.

> **Documentation Structure**: This project uses specialized AI instructions for agents and human-readable guides in [docs/](../../docs/). See also [Learning Objectives](learning-objectives.instructions.md) and [Showcase Criteria](showcase-criteria.instructions.md) for context.

## Key Architectural Decisions
- **Package Manager**: uv (only)
- **Messaging**: `confluent-kafka` for all services (core infrastructure)
- **Deployment Modes**: Training (in-memory) vs Production (Kafka distributed)
- **Configuration**: YAML with Pydantic validation (see [ADRs](../../docs/adr/ai-agent-context.md))
- **Strategy Isolation**: Framework open-source, strategies proprietary (see [README](../../README.md))

## Dependency Management
Use **project-level** dependency groups in each service's pyproject.toml:

```toml
[dependency-groups]
test = ["pytest>=8.0.0", "pytest-cov>=4.0.0", "pytest-mock>=3.12.0"]
dev = ["mypy>=1.8.0", "ruff>=0.1.0", "ipython>=8.0.0"]
dev-full = ["mypy>=1.8.0", "pytest>=8.0.0", "..."] # Combined dev + test
```

### Standard Versions (maintain consistency)
- pytest>=8.0.0, mypy>=1.8.0, ruff>=0.1.0, PyYAML>=6.0

### Commands
```bash
# Development setup
uv sync --group dev-full

# Service-specific
cd drl-trading-{service} && uv sync --group dev-full

# Production
cd drl-trading-{service} && uv sync
```

## Service Structure Standards
All services must follow this standardized structure (implemented via T004):

### Directory Structure (FINAL IMPLEMENTATION)
```
drl-trading-{service}/
├── config/                        # YAML configuration files (root level for EnhancedServiceConfigLoader)
│   ├── application.yaml           # Base configuration
│   ├── application-local.yaml     # Local development overrides
│   ├── application-cicd.yaml      # CI/CD environment overrides
│   └── application-prod.yaml      # Production environment overrides
├── .env                           # Local development environment variables
├── main.py                        # Standardized entry point (minimal bootstrap)
├── src/
│   └── drl_trading_{service}/
│       ├── adapter/               # Hexagonal: External adapters
│       │   ├── web/              # REST API adapters (if applicable)
│       │   ├── cli/              # Command-line adapters (if applicable)
│       │   └── messaging/        # Message bus adapters (if applicable)
│       ├── core/                  # Hexagonal: Business logic
│       │   ├── port/             # Hexagonal: Interfaces/contracts
│       │   ├── service/          # Hexagonal: Business services
│       │   └── model/            # Domain entities
│       └── infrastructure/       # Hexagonal: Technical concerns
│           ├── bootstrap/        # Service bootstrap logic
│           │   └── {service}_function_bootstrap.py
│           ├── config/           # Python configuration classes only
│           │   ├── {service}_config.py
│           │   └── __init__.py
│           └── di/               # Dependency injection modules
│               └── {Service}Module.py
├── tests/
│   ├── integration/
│   └── unit/
├── docker/
│   └── Dockerfile
└── pyproject.toml
```

### Entry Point Standardization (T004 Compliant)
All services use identical main.py pattern:

```python
# main.py (standardized across all services)
"""
Main entry point for DRL Trading {Service} Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_{service}.infrastructure.bootstrap.{service}_function_bootstrap import bootstrap_{service}_service


def main() -> None:
    """
    Main entry point for {service} service.

    Uses standardized bootstrap pattern while maintaining
    hexagonal architecture compliance.
    """
    bootstrap_{service}_service()


if __name__ == "__main__":
    main()
```

### Bootstrap Pattern (Function-Based)
All services implement standardized function-based bootstrap:

```python
# src/drl_trading_{service}/infrastructure/bootstrap/{service}_function_bootstrap.py
"""Function-based bootstrap for {service} service following T004 patterns."""
import logging
from typing import Optional

from drl_trading_common.config.logging_config import configure_unified_logging
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader
from drl_trading_{service}.infrastructure.config.{service}_config import {Service}Config

def bootstrap_{service}_service() -> None:
    """Bootstrap the {service} service with T004 compliance."""
    # 1. Load configuration via EnhancedServiceConfigLoader
    # 2. Setup unified logging
    # 3. Initialize service-specific logic
    # 4. Handle graceful shutdown
```

### Hexagonal Architecture Standards
All deployable services MUST follow hexagonal architecture:

#### Core Layer (`core/`)
- **Business Logic**: Domain services and entities
- **Ports (`port/`)**: Interfaces defining contracts (repositories, services)
- **Services (`service/`)**: Business service implementations
- **No Dependencies**: Core must not depend on external frameworks

#### Adapter Layer (`adapter/`)
- **External Interfaces**: REST APIs, CLI, message handlers, database adapters
- **Implementation**: Concrete implementations of core ports
- **Framework Integration**: FastAPI, Click, SQLAlchemy adapters

#### Infrastructure Layer (`infrastructure/`)
- **Technical Concerns**: Configuration, dependency injection, bootstrapping
- **Cross-cutting**: Logging, monitoring, security
- **Bootstrap (`bootstrap/`)**: Application startup and wiring

### Configuration Standards (T004 Implementation)
- **YAML Location**: Root `config/` directory (required for EnhancedServiceConfigLoader compatibility)
- **Python Classes**: `src/drl_trading_{service}/infrastructure/config/` directory
- **STAGE Environment Variable**: Controls config file selection (local/cicd/prod)
- **Base Config**: All services inherit from `BaseApplicationConfig`
- **Common Components**: `MessagingConfig`, `LoggingConfig` (required), `InfrastructureConfig`
- **Optional Components**: `DatabaseConfig` (service-dependent)
- **Service-Specific**: Custom config objects per service needs

### Configuration File Pattern:
```yaml
# config/application.yaml (base configuration)
app_name: "drl-trading-{service}"
stage: "development"
version: "1.0.0"

infrastructure:
  messaging:
    provider: "in_memory"  # in_memory for training, kafka for production
  logging:
    level: "INFO"
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: "logs/{service}.log"
    console_enabled: true

# Service-specific configuration sections...
```

```yaml
# config/application-local.yaml (local overrides)
infrastructure:
  messaging:
    provider: "in_memory"
  database:
    host: "localhost"
    port: 5432
# Local development overrides...
```

### Python Configuration Classes:
```python
# src/drl_trading_{service}/infrastructure/config/{service}_config.py
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig

class {Service}Config(BaseApplicationConfig):
    """T004-compliant configuration for {service} service."""

    infrastructure: InfrastructureConfig
    # Service-specific config objects...
```

### Environment Management
- **Local**: `.env` file with STAGE=local
- **CI/CD**: Environment variables via GitLab CI
- **Production**: AWS SSM Parameter Store
- **Config Paths**: Must be constant across environments
