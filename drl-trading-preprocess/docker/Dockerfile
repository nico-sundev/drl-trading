# =============================================================================
# Builder stage - Install dependencies and build project (non-editable)
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy workspace-level lockfile and root pyproject.toml first
COPY uv.lock pyproject.toml ./

# Copy workspace member pyproject.toml files (needed for workspace structure, not dependency resolution)
COPY drl-trading-common/pyproject.toml ./drl-trading-common/
COPY drl-trading-core/pyproject.toml ./drl-trading-core/
COPY drl-trading-adapter/pyproject.toml ./drl-trading-adapter/
COPY drl-trading-strategy-example/pyproject.toml ./drl-trading-strategy-example/
COPY drl-trading-preprocess/pyproject.toml ./drl-trading-preprocess/

# Install dependencies only (leverages Docker layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    cd drl-trading-preprocess && \
    uv sync --frozen --no-install-workspace --no-dev

# Now copy the actual source code for workspace packages
COPY drl-trading-common ./drl-trading-common
COPY drl-trading-core ./drl-trading-core
COPY drl-trading-adapter ./drl-trading-adapter
COPY drl-trading-strategy-example ./drl-trading-strategy-example
COPY drl-trading-preprocess ./drl-trading-preprocess

# Install workspace packages (non-editable for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    cd drl-trading-preprocess && \
    uv sync --frozen --no-dev --no-editable

# =============================================================================
# Runtime stage - Minimal production image
# =============================================================================
FROM python:3.12-slim-bookworm

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r trading && \
    useradd -r -g trading -u 1000 trading && \
    mkdir -p /app && \
    chown -R trading:trading /app

WORKDIR /app

# Copy virtual environment from builder (using /opt/venv from UV_PROJECT_ENVIRONMENT)
COPY --from=builder --chown=trading:trading /opt/venv /opt/venv

# Copy workspace packages needed at runtime (installed as non-editable, so we need the source)
COPY --from=builder --chown=trading:trading /app/drl-trading-common /app/drl-trading-common
COPY --from=builder --chown=trading:trading /app/drl-trading-core /app/drl-trading-core
COPY --from=builder --chown=trading:trading /app/drl-trading-adapter /app/drl-trading-adapter
COPY --from=builder --chown=trading:trading /app/drl-trading-strategy-example /app/drl-trading-strategy-example
COPY --from=builder --chown=trading:trading /app/drl-trading-preprocess /app/drl-trading-preprocess

# Copy only runtime files needed
COPY --chown=trading:trading drl-trading-preprocess/main.py ./main.py
COPY --chown=trading:trading drl-trading-preprocess/config ./config

# Create required directories
RUN mkdir -p /app/data/feature_store /app/logs && \
    chown -R trading:trading /app/data /app/logs

# Switch to non-root user
USER trading

# Activate virtual environment by putting it first in PATH
# Set PYTHONPATH to include workspace packages
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/drl-trading-common/src:/app/drl-trading-core/src:/app/drl-trading-adapter/src:/app/drl-trading-strategy-example/src:/app/drl-trading-preprocess/src:$PYTHONPATH" \
    STAGE=local

# Health check (using correct port 8082 and curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Run service
CMD ["python", "main.py"]
