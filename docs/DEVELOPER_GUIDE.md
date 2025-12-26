# Developer Guide

This guide covers development standards, testing practices, and coding workflows for contributing to the DRL Trading Framework.

## Development Standards

- **Code Quality**: All code must pass `ruff check`, `mypy`, and `pytest`
- **Testing**: Follow Given/When/Then structure for all tests
- **Architecture**: Follow hexagonal architecture and SOLID principles
- **Documentation**: Update relevant docs with architectural decisions

## Service Structure

All services follow hexagonal architecture:
```
src/drl_trading_{service}/
├── adapter/           # External interfaces
├── core/
│   ├── port/         # Business contracts
│   └── service/      # Business logic
└── application/
│   └── config/      # Config classes
│   └── di/      # Injector module for DI
```

**Key principles:**
- **Core**: Business logic, no external dependencies
- **Adapters**: External interfaces (Kafka, REST APIs, databases)
- **Application**: Configuration, dependency injection, main entry point

**→ See [System Architecture](SYSTEM_ARCHITECTURE.md)** for detailed architecture decisions.

## Development Workflow

### Local Development Setup

**→ See [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md#local-development-setup)** for environment setup.

### Development Cycle

**1. Start Infrastructure:**
```bash
./infrastructure/local/docker_compose/run-services.sh infra
```

**2. Develop Service Locally:**
```bash
cd drl-trading-preprocess
uv sync --group dev-full
STAGE=local uv run python main.py
```

**3. Run Tests:**
```bash
# Unit tests
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Specific test file
uv run pytest tests/unit/data_set_utils/merge_service_test.py -v
```

**4. Code Quality Checks:**
```bash
# Linting and formatting
ruff check . --fix
ruff format .

# Type checking
mypy src/

# Run all checks before committing
ruff check . && mypy src/ && pytest tests/
```

## Testing Standards

### Test Organization

```
tests/
├── unit/              # Unit tests (no external dependencies)
│   ├── core/         # Business logic tests
│   └── adapter/      # Adapter tests (with mocks)
├── integration/       # Integration tests (with real dependencies)
│   ├── database/     # Database integration
│   └── kafka/        # Messaging integration
└── resources/         # Test data and fixtures
```

### Test Structure (Given/When/Then)

**ALL test methods MUST follow this structure:**

```python
def test_merge_datasets_removes_duplicates(self) -> None:
    """Test that merge service removes duplicate entries."""
    # Given
    dataset1 = pd.DataFrame({"symbol": ["BTC"], "timestamp": [100]})
    dataset2 = pd.DataFrame({"symbol": ["BTC"], "timestamp": [100]})

    # When
    result = merge_service.merge(dataset1, dataset2)

    # Then
    assert len(result) == 1
    assert result["symbol"].iloc[0] == "BTC"
```

### Test Coverage Requirements

- **Target**: 80%+ code coverage
- **Priority**: Cover critical business logic first
- **Tools**: `pytest-cov` for coverage reports

```bash
# Generate coverage report
uv run pytest --cov=src/drl_trading_preprocess --cov-report=html

# View report
open htmlcov/index.html
```

### Test Types

**Unit Tests:**
- Mock all external dependencies
- Test single function/method behavior
- Fast execution (< 1s per test)
- Location: `tests/unit/`

**Integration Tests:**
- Use real external dependencies (databases, Kafka)
- Test adapter implementations and service interactions
- Slower execution (few seconds per test)
- Location: `tests/integration/`

**Service E2E Tests:**
- Test complete workflows within a single service
- Validate full service behavior with real external systems
- Example: `production_training_workflow_e2e_test.py` validates data ingestion → resampling → feature computation → Feast persistence
- Run with Docker infrastructure (Kafka, TimescaleDB, Feast)
- Location: `tests/e2e/`

**System E2E Tests:** *(Not yet implemented)*
- Test complete workflows across multiple services
- Validate full system behavior (data ingestion → preprocessing → training → inference → execution)
- Requires all services running in Docker containers
- **→ See [Infrastructure Guide - E2E Testing](INFRASTRUCTURE_GUIDE.md#end-to-end-testing)** for setup and execution

**Architecture Tests:**
- Enforce hexagonal architecture rules
- Use `import-linter` to prevent violations

```bash
# Run architecture tests
pytest tests/unit/architecture/ -v
pytest tests/architecture_all_services_test.py -v
```

## Code Quality Standards

- **Testing**: Given/When/Then structure, 80%+ coverage
- **Quality**: `ruff check . --fix` and `mypy src/`
- **Architecture**: Hexagonal patterns, SOLID principles
  - **Architecture Tests**: Automated enforcement via import-linter (see [tests/README.md](../tests/README.md))
  - Run per-service: `pytest tests/unit/architecture/ -v`
  - Run all services: `pytest tests/architecture_all_services_test.py -v`
- **Dependencies**: uv only, project-level groups

## Configuration Best Practices

Services use environment-specific YAML:
- `config/application-local.yaml`
- `config/application-cicd.yaml`
- `config/application-prod.yaml`

**Configuration guidelines:**
- Never hardcode credentials
- Use environment variables for secrets
- Keep config files in version control (except secrets)
- Document all configuration options

## Troubleshooting Development Issues

**→ See [Infrastructure Guide - Troubleshooting](INFRASTRUCTURE_GUIDE.md#troubleshooting)** for common issues and solutions.

---

**→ Next Steps:**
- **Setup infrastructure**: [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **Understand architecture**: [System Architecture](SYSTEM_ARCHITECTURE.md)
- **Develop strategies**: [Strategy Development](STRATEGY_DEVELOPMENT.md)
