FROM python:3.11-slim-bookworm

WORKDIR /app

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies
RUN uv sync

# Set environment
ENV PYTHONPATH=/app/src

# Run service
CMD ["uv", "run", "main.py"]
