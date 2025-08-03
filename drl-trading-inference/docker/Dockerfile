FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies
RUN uv sync

# Set environment
ENV PYTHONPATH=/app/src

# Run service
CMD ["uv", "run", "python", "-m", "drl_trading_inference.bootstrap"]
