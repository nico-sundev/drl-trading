# Makefile for DRL Trading Monorepo
.PHONY: help install test lint format clean generate-clients ci-setup ci-test ci-build

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies for all services"
	@echo "  test             - Run tests for all services"
	@echo "  lint             - Run linting for all services"
	@echo "  format           - Format code for all services"
	@echo "  clean            - Clean build artifacts"
	@echo "  generate-clients - Generate OpenAPI clients"
	@echo "  ci-setup         - Setup for CI environment"
	@echo "  ci-test          - Run tests in CI environment"
	@echo "  ci-build         - Build for CI environment"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@for dir in drl-trading-*; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Installing $$dir..."; \
			cd $$dir && uv sync --group dev-full && cd ..; \
		fi; \
	done

# Run tests
test:
	@echo "Running tests..."
	@for dir in drl-trading-*; do \
		if [ -d "$$dir/tests" ]; then \
			echo "Testing $$dir..."; \
			cd $$dir && uv run pytest tests/ -v --cov=src --cov-report=html && cd ..; \
		fi; \
	done

# Run linting
lint:
	@echo "Running linting..."
	@for dir in drl-trading-*; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Linting $$dir..."; \
			cd $$dir && uv run ruff check src/ && cd ..; \
		fi; \
	done

# Format code
format:
	@echo "Formatting code..."
	@for dir in drl-trading-*; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Formatting $$dir..."; \
			cd $$dir && uv run ruff check src/ --fix && cd ..; \
		fi; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name "dist" -exec rm -rf {} +
	@find . -type d -name "build" -exec rm -rf {} +

# Generate OpenAPI clients
generate-clients:
	@echo "Generating OpenAPI clients..."
	@./scripts/generate-and-install-clients.sh

# CI setup
ci-setup:
	@echo "Setting up CI environment..."
	@pip install uv
	@uv sync --group dev-full

# CI test
ci-test:
	@echo "Running CI tests..."
	@uv run pytest tests/ -v --cov=src --cov-report=xml:coverage.xml --cov-report=html

# CI build
ci-build:
	@echo "Building for CI..."
	@uv run python -m build
