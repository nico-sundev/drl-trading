# Docker Compose - Centralized Infrastructure

This directory contains the centralized Docker Compose configuration for the entire AI Trading system.

## Architecture Decision

**Centralized compose with profiles** - Best practice for monorepo microservices:
- ✅ Single source of truth for shared infrastructure
- ✅ Selective service startup via profiles
- ✅ No infrastructure duplication
- ✅ Easy E2E testing across services

## Quick Start

### Infrastructure Only
```bash
# Linux/Mac
./run-services.sh infra

# Windows
run-services.bat infra
```

### With Specific Service
```bash
# Preprocess service + infrastructure
./run-services.sh preprocess

# Ingest service + infrastructure
./run-services.sh ingest
```

### All Services
```bash
./run-services.sh all
```

### Stop Everything
```bash
./run-services.sh down
```

## Available Profiles

| Profile | Services Started |
|---------|------------------|
| (none) | Infrastructure only: Kafka, Zookeeper, TimescaleDB, Postgres |
| `preprocess` | Infrastructure + Preprocess service |
| `ingest` | Infrastructure + Ingest service |
| `all` | Everything (all services) |

## Infrastructure Components

### Kafka + Zookeeper
- **Kafka**: `localhost:9092` (host) / `kafka:29092` (container-to-container)
- **Zookeeper**: `localhost:2181`
- **Topics**: Auto-created via `kafka-init` service
  - `ready.rawdata.batch`
  - `ready.rawdata.increment`
  - `requested.preprocess-data`
  - `requested.store-resampled-data`
  - `completed.preprocess-data`
  - `error.preprocess-data`

### TimescaleDB
- **Host**: `localhost:5432`
- **Database**: `marketdata`
- **User/Pass**: `postgres/postgres`

### Feature Config DB (Postgres)
- **Host**: `localhost:5431`
- **Database**: `feature_config_db`
- **User/Pass**: `fc_user/fc_pass`

## Manual Docker Compose Commands

If you prefer manual control:

```bash
# Infrastructure only
docker-compose up

# With preprocess service
docker-compose --profile preprocess up --build

# With ingest service
docker-compose --profile ingest up --build

# All services
docker-compose --profile all up --build

# Detached mode
docker-compose --profile preprocess up -d

# Stop all
docker-compose --profile all down

# View logs
docker-compose logs -f preprocess-service
```

## Adding New Services

To add a new microservice to the compose file:

1. Add service definition with appropriate profile:
```yaml
  your-service:
    build:
      context: ..
      dockerfile: drl-trading-your-service/docker/Dockerfile
    profiles: ["your-service", "all"]
    ports:
      - "8081:8081"
    environment:
      - STAGE=local
    depends_on:
      - kafka
```

2. Update `run-services.sh` and `run-services.bat` with new profile option

## Development Workflow

### Typical Flow
```bash
# 1. Start infrastructure
./run-services.sh infra

# 2. Develop service locally (outside Docker)
cd ../drl-trading-preprocess
STAGE=local uv run python main.py

# 3. Test E2E with Dockerized service
./run-services.sh preprocess

# 4. Clean up
./run-services.sh down
```

### Rebuilding After Code Changes
```bash
# Rebuild and restart specific service
docker-compose --profile preprocess up --build

# Force rebuild (ignore cache)
docker-compose --profile preprocess build --no-cache
docker-compose --profile preprocess up
```

## Troubleshooting

### Kafka Connection Issues
- Ensure you use `kafka:29092` from within containers
- Use `localhost:9092` from host machine

### Port Conflicts
- TimescaleDB: 5432
- Feature DB: 5431
- Kafka: 9092
- Zookeeper: 2181
- Services: 8080+ (check each service's config)

### Volume Cleanup
```bash
# Remove all volumes (data loss!)
docker-compose down -v
```

## Best Practices

1. **Always use profiles** to avoid starting unnecessary services
2. **Use `--build` flag** when testing code changes
3. **Check logs** with `docker-compose logs -f <service-name>`
4. **Mount volumes** for logs/config during development (already configured)
5. **Stop services** with `./run-services.sh down` (not Ctrl+C) for clean shutdown

## CI/CD Integration

For CI pipelines, use profiles to test specific services:

```yaml
# Example GitHub Actions
- name: Start test infrastructure
  run: docker-compose up -d

- name: Run E2E tests for preprocess
  run: docker-compose --profile preprocess up --build --abort-on-container-exit
```
