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
All services must follow this standardized structure:

### Directory Structure
```
drl-trading-{service}/
├── config/
│   ├── application-local.yaml
│   ├── application-cicd.yaml
│   └── application-prod.yaml
├── .env                           # Local development only
├── src/
│   └── drl_trading_{service}/
│       ├── bootstrap.py           # Service entrypoint
│       ├── {service}Module.py     # DI module
│       ├── adapter/               # Hexagonal: External adapters
│       ├── core/                  # Hexagonal: Business logic
│       │   ├── port/             # Hexagonal: Interfaces/contracts
│       │   └── service/          # Hexagonal: Business services
│       └── infrastructure/       # Hexagonal: Technical concerns
│           ├── bootstrap/
│           ├── config/
│           └── di/
├── tests/
│   ├── integration/
│   └── unit/
├── docker/
│   └── Dockerfile
└── pyproject.toml
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

### Configuration Standards
- **STAGE Environment Variable**: Controls config file selection (local/cicd/prod)
- **Base Config**: All services inherit from `BaseApplicationConfig`
- **Common Components**: `MessagingConfig`, `LoggingConfig` (required)
- **Optional Components**: `DatabaseConfig` (service-dependent)
- **Service-Specific**: Custom config objects per service needs

### Environment Management
- **Local**: `.env` file with STAGE=local
- **CI/CD**: Environment variables via GitLab CI
- **Production**: AWS SSM Parameter Store
- **Config Paths**: Must be constant across environments
