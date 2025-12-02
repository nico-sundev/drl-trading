#!/bin/bash
# Script to sync dependencies with artifactory sources if configured in .env

# Source .env if it exists
if [ -f .env ]; then
    source .env
    echo "Sourced .env"
    echo "UV_SOURCES__DRL_TRADING_STRATEGY_EXAMPLE=$UV_SOURCES__DRL_TRADING_STRATEGY_EXAMPLE"
fi

# Run uv sync
uv sync "$@"
