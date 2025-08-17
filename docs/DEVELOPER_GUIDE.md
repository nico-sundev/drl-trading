# Developer Guide

> **For AI Agents**: See [.github/instructions/](./.github/instructions/)

## Quick Setup

```bash
# Clone and install
git clone https://github.com/yourusername/drl-trading.git
cd drl-trading
uv sync --group dev-full

# Verify setup
./scripts/verify_setup.sh
```

## Service Development

All services follow [hexagonal architecture](ARCHITECTURE.md):
```
src/drl_trading_{service}/
├── adapter/           # External interfaces
├── core/
│   ├── port/         # Business contracts
│   └── service/      # Business logic
└── infrastructure/   # Config, DI, bootstrap
```

## Development Workflow

```bash
# Service development
cd drl-trading-{service}
uv sync --group dev-full
uv run pytest tests/
uv run ruff check . --fix
uv run mypy src/

# Startup observability
# Each service startup is instrumented with StartupContext.
# Wrap new bootstrap steps in ctx.phase("<name>") and add dependency probes instead of ad-hoc logs.
# See OBSERVABILITY_STARTUP.md for schema & conventions.

# Integration testing
docker-compose -f docker-compose.training.yml up
./scripts/e2e_test.sh
```

## Code Standards

- **Testing**: Given/When/Then structure, 80%+ coverage
- **Quality**: `ruff check . --fix` and `mypy src/`
- **Architecture**: Hexagonal patterns, SOLID principles
- **Startup Observability**: Maintain `STARTUP SUMMARY` schema (see OBSERVABILITY_STARTUP.md); add phases sparingly; prefer attributes & probes.
- **Dependencies**: uv only, project-level groups

## Configuration

Services use environment-specific YAML:
- `config/application-local.yaml`
- `config/application-cicd.yaml`
- `config/application-prod.yaml`

## Troubleshooting

```bash
# Dependency issues
uv cache clean && rm uv.lock && uv sync --group dev-full

# Test discovery
uv sync --group test && uv run pytest --collect-only

# Docker issues
docker system prune -a && docker-compose build --no-cache
```

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details.*
