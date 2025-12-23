# CI/CD Image: UV + Docker + Python 3.12
# This image combines UV package manager with Docker CLI for integration tests
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

LABEL maintainer="drl-trading"
LABEL description="UV Python package manager with Docker CLI for CI/CD"

# Install Docker CLI and essential tools
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    git \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN uv --version && docker --version

# Set working directory
WORKDIR /workspace

# Default command (can be overridden by GitLab CI)
CMD ["/bin/bash"]
