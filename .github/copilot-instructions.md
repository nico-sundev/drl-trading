# AI Coding Agent Instructions - DRL Trading Framework

## Project Architecture

**Hexagonal Architecture Pattern** - ALL services follow strict layer separation:
```
src/drl_trading_{service}/
├── adapter/        # External interfaces (REST, Kafka, DB)
├── core/
│   ├── port/      # Business contracts (interfaces)
│   └── service/   # Business logic (framework-agnostic)
└── infrastructure/
    ├── bootstrap/ # Service startup
    ├── config/    # Python config classes
    └── di/        # Dependency injection modules
```

**Critical Rules:**
- Core layer NEVER depends on infrastructure/adapter layers
- Dependencies always point inward (Infrastructure → Core)
- Business logic communicates via port interfaces only
- Adapters implement ports and inject via DI (using `injector` library)

**Example DI Wiring:**
```python
# Core service depends on port interface
class MarketDataService:
    def __init__(self, reader: MarketDataReaderPort): ...

# DI module binds port to adapter
binder.bind(MarketDataReaderPort, to=TimescaleMarketDataRepo, scope=singleton)
```

## Event-Driven Messaging (Kafka)

**Topic-Based Routing** - Configuration-driven handler mapping:
```yaml
# config/application.yaml
infrastructure:
  kafka:
    consumer:
      topic_subscriptions:
        - topic: "requested.preprocess-data"
          handler_id: "preprocessing_request_handler"
```

**Handler Pattern** - Use `@kafka_handler` decorator:
```python
@kafka_handler(FeaturePreprocessingRequest, "preprocessing_request_handler")
def handle_request(request: FeaturePreprocessingRequest, message: Message) -> None:
    # Deserialize → Delegate to core service → Exceptions propagate for retry/DLQ
    orchestrator.process_feature_computation_request(request)
```

**Key Points:**
- Handlers registered in DI modules, mapped to topics in YAML
- NO manual offset commits (adapter handles)
- NO business logic in handlers (delegate to core services)
- Exceptions trigger retry/DLQ based on failure policies

## Configuration Management

**YAML-First with Pydantic Validation:**
```
config/
├── application.yaml           # Base config
├── application-local.yaml     # Local dev overrides
├── application-ci.yaml        # CI/CD overrides
└── application-prod.yaml      # Production overrides
```

**Loading Pattern:**
```python
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

config = EnhancedServiceConfigLoader.load_config(
    InferenceConfig,
    service="inference",
    env_override=True  # Enables STAGE-based override file selection
)
```

**Environment Variable Overrides:** Use `STAGE` environment variable to select override file (e.g., `STAGE=ci` loads `application-ci.yaml`).

## Testing Requirements

**ALL tests MUST follow Given/When/Then structure with explicit comments:**
```python
def test_feature_computation(self, mock_reader: Mock) -> None:
    """Test RSI computation with valid OHLCV data."""
    # Given
    ohlcv_data = create_sample_ohlcv(periods=50)
    mock_reader.fetch_data.return_value = ohlcv_data

    # When
    result = service.compute_features(symbol="BTCUSD", feature="rsi")

    # Then
    assert len(result) == 50
    assert "rsi_14" in result.columns
    mock_reader.fetch_data.assert_called_once_with("BTCUSD", timeframe="1m")
```

**Test Organization:**
- Unit tests: `tests/unit/` (mocked dependencies)
- Integration tests: `tests/integration/` (real implementations, may use files)
- E2E tests: `tests/e2e/` (real Kafka/DB, running service)
- Test file naming: `{module}_test.py` (unit), `{module}_it.py` (integration)
- Fixtures in `conftest.py` for reuse across test files
- Use test data builders for complex objects

**Coverage Requirements:** Near 100% coverage. Use `#runTests` tool or `uv run python -m pytest`.

## Development Workflow

**Dependency Management:** Always use `uv sync --group dev-all` (never just `uv sync`)

**Code Quality Pipeline:**
```bash
# 1. Fix formatting/linting
uv run ruff check <file> --fix

# 2. Validate types using #problems tool

# 3. Run tests
uv run pytest tests/unit/  # Fast feedback
uv run pytest tests/e2e/   # Requires Docker Compose
```

**Service Bootstrap Pattern:**
```python
# main.py (ALL services use this pattern)
from drl_trading_{service}.infrastructure.bootstrap.{service}_service_bootstrap import {Service}ServiceBootstrap

def main() -> None:
    bootstrap = {Service}ServiceBootstrap()
    bootstrap.start()  # Handles config, DI, logging, Kafka consumers
```

**Database Migrations:** Owned by `drl-trading-ingest` service using Alembic. Other services READ via `MarketDataReaderPort`.

## E2E Testing Critical Knowledge

**Prerequisites:** Docker Compose + initialized database schema
```bash
# Automated (recommended)
./scripts/run_e2e_tests.sh

# Manual setup
docker-compose -f docker_compose/docker-compose.yml up -d
./scripts/init_e2e_database.sh  # Creates market_data table via ingest migrations
STAGE=ci uv run python main.py  # Start service
pytest tests/e2e/ -v
```

**Service State Management:** Preprocess service maintains resampling state (`state/resampling_context.json`). Restart service or delete state file between test runs.

**Fixture Pattern:**
```python
def test_workflow(publish_kafka_message, kafka_consumer_factory, wait_for_kafka_message):
    # Given - Create consumer BEFORE publishing
    consumer = kafka_consumer_factory(['output.topic'])
    time.sleep(2)  # Let consumer subscribe

    # When - Publish to service
    publish_kafka_message(topic='input.topic', key='test', value={...})

    # Then - Wait for output
    result = wait_for_kafka_message(consumer, timeout=30)
```

## Common Anti-Patterns to Avoid

❌ Core services depending on infrastructure (violates hexagonal architecture)
❌ Business logic in Kafka handlers (handlers should only deserialize → delegate)
❌ Mocking Kafka in E2E tests (use real Kafka)
❌ Test methods without Given/When/Then comments
❌ Creating class instances directly instead of using DI
❌ Forgetting to restart service between E2E test runs (stale resampling state)

## Intellectual Rigor Expected

When receiving instructions to implement features:
1. **Challenge design fit** - Does this belong in core vs adapter vs infrastructure?
2. **Evaluate DI injection** - Should this be singleton? Request-scoped?
3. **Consider shared data** - Does this belong in `drl-trading-common`?
4. **Review existing patterns** - Check for similar implementations before creating new ones
5. **Self-review as senior developer** - Validate SOLID principles, test coverage, documentation
6. **Express architectural concerns** - If design violates patterns, explain with specific examples

**Response Style:** Keep responses concise. After completing work, end with "Done. What are we tackling next?" - NO verbose summaries.

## Key Files for Reference

- **ADRs:** `docs/adr/` - Architectural decisions with rationale
- **Hexagonal structure:** `drl-trading-ingest/src/` - Reference implementation
- **Kafka patterns:** `drl-trading-common/src/drl_trading_common/adapter/messaging/`
- **Config architecture:** `docs/adr/0002-configuration-architecture-standardization.md`
- **E2E testing:** `drl-trading-preprocess/tests/e2e/README.md`
- **Market data access:** `docs/MARKET_DATA_SHARED_ACCESS.md`

## Project-Specific Conventions

- **Package manager:** `uv` (not pip/poetry)
- **Python version:** 3.12+
- **DI framework:** `injector` library with module-based configuration
- **Message broker:** Kafka with topic-based routing (no handler_id headers)
- **Database:** TimescaleDB for time-series, Feast for feature store
- **Type checking:** Type hints mandatory on all functions/methods
- **Docstrings:** Required for all classes and public methods
- **Configuration:** YAML files in root `config/`, Python classes in `infrastructure/config/`
