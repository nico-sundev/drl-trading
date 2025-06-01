# Multi-stage Dockerfile for both training and production

# Base stage with common dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY drl-trading-core/pyproject.toml ./framework/pyproject.toml
COPY drl-trading-impl-example/pyproject.toml ./impl/pyproject.toml

# Install Python dependencies
RUN pip install -e ./framework/
RUN pip install -e ./impl/

# Copy source code
COPY drl-trading-core/src ./framework/src
COPY drl-trading-impl-example/src ./impl/src
COPY drl-trading-impl-example/config ./impl/config

# Training stage
FROM base as training

ENV DEPLOYMENT_MODE=training
ENV DRL_TRADING_CONFIG_PATH=/app/impl/config/applicationConfig.json

WORKDIR /app/impl

CMD ["python", "main.py"]

# Production stage
FROM base as production

# Install production dependencies (RabbitMQ client)
RUN pip install pika

ENV DEPLOYMENT_MODE=production
ENV DRL_TRADING_CONFIG_PATH=/app/impl/config/applicationConfig.json
ENV RABBITMQ_HOST=rabbitmq
ENV RABBITMQ_PORT=5672

# Create different entry points for different services
COPY scripts/production/ ./scripts/

# Default to data ingestion service, but can be overridden
CMD ["python", "scripts/data_ingestion_service.py"]
