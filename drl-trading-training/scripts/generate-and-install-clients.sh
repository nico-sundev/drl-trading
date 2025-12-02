#!/bin/bash
# CI/CD script for generating and installing OpenAPI clients

set -e  # Exit on any error

echo "🔄 Generating OpenAPI clients..."

# Generate the client
uv run generate-clients

# Install the generated client package
echo "📦 Installing generated client package..."
cd src/drl_trading_training/adapter/rest/generated
uv pip install -e .

echo "✅ Client generation and installation complete"
