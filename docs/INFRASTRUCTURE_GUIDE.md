# Infrastructure Guide

This guide covers infrastructure setup, deployment, and operational concerns for the DRL Trading Framework.

## Local Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/nico-sundev/drl-trading.git
cd drl-trading

# Install dependencies for all services
uv sync --group dev-full

# Generate OpenAPI clients
./scripts/openapi/generate-and-install-clients.sh

# Start infrastructure (PostgreSQL, Kafka, Redis)
./infrastructure/local/docker_compose/run-services.sh infra
```

### Infrastructure Components

The local environment includes:

| Component | Port | Purpose |
|-----------|------|---------|
| **TimescaleDB** | 5432 | Time-series OHLCV data storage |
| **Kafka** | 9092 | Event bus for service communication |
| **Zookeeper** | 2181 | Kafka coordination |
| **PostgreSQL** (Feature Config) | 5431 | Feast feature registry |
| **Redis** | 6379 | Feast online feature store |

### Starting Individual Services

```bash
# Infrastructure only
./infrastructure/local/docker_compose/run-services.sh infra

# Infrastructure + preprocess service
docker-compose --profile preprocess up

# Infrastructure + specific service
docker-compose --profile ingest up
docker-compose --profile training up

# Everything
docker-compose --profile all up
```

### Running Services Locally (Outside Docker)

For development, run services outside Docker for faster iteration:

```bash
# Terminal 1: Start infrastructure
./infrastructure/local/docker_compose/run-services.sh infra

# Terminal 2: Run preprocess service locally
cd drl-trading-preprocess
uv sync --group dev-full
STAGE=local uv run python main.py

# Terminal 3: Run another service
cd drl-trading-ingest
STAGE=local uv run python main.py
```

**Benefits:**
- Faster code reload (no Docker rebuild)
- Direct access to debugger
- Easier log inspection

## Environment Configuration

Services use environment-specific YAML configuration:

```
service/
├── config/
│   ├── application-local.yaml    # Local development
│   ├── application-cicd.yaml     # CI/CD pipeline
│   └── application-prod.yaml     # Production deployment
```

**Selecting environment:**
```bash
STAGE=local uv run python main.py   # Uses application-local.yaml
STAGE=cicd uv run python main.py    # Uses application-cicd.yaml
STAGE=prod uv run python main.py    # Uses application-prod.yaml
```

## CI/CD Setup

### GitLab CI/CD with Docker Support

The project uses a custom GitLab CI image with Docker-in-Docker support for integration tests.

**→ See [CI Image Setup Guide](CI_IMAGE_SETUP.md)** for detailed AWS and GitLab configuration.

## End-to-End Testing (TODO)

### Running E2E Tests

```bash
# Start all services
docker-compose -f docker-compose.training.yml up -d

# Wait for services to be healthy
sleep 30

# Run E2E test script
./scripts/e2e_test.sh

# Check logs
docker-compose logs -f preprocess-service
docker-compose logs -f training-service
```

### E2E Test Scenarios

The `e2e_test.sh` script validates:
1. **Data ingestion**: Historical data fetch from data provider
2. **Preprocessing**: Feature computation and Feast storage
3. **Training**: Model training with example strategy
4. **Model registration**: Artifact storage in MLflow

## Troubleshooting

### Common Issues

#### Dependency Problems

```bash
# Clean uv cache
uv cache clean

# Remove lock file and resync
rm uv.lock
uv sync --group dev-full
```

#### Docker Issues

```bash
# Clean everything (⚠️ removes all containers/images)
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check container logs
docker-compose logs -f <service-name>
```

## AWS Deployment (TODO)

*This section will cover production deployment on AWS, including:*

- ECS/EKS service deployment
- RDS for PostgreSQL
- MSK for Kafka
- ElastiCache for Redis
- S3 for Feast offline store
- CloudWatch for monitoring

---

**→ Next Steps:**
- **Development standards**: [Developer Guide](DEVELOPER_GUIDE.md)
- **System architecture**: [System Architecture](SYSTEM_ARCHITECTURE.md)
- **Strategy development**: [Strategy Development](STRATEGY_DEVELOPMENT.md)
