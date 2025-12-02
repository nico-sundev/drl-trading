#!/bin/bash
# Generate and install OpenAPI clients for DRL Trading services

set -e

echo "Generating OpenAPI clients..."

# Find the ingest service spec
INGEST_SPEC=""
if [ -f "drl-trading-ingest/specs/openapi.yaml" ]; then
    INGEST_SPEC="drl-trading-ingest/specs/openapi.yaml"
elif [ -f "drl-trading-ingest/specs/openapi.yml" ]; then
    INGEST_SPEC="drl-trading-ingest/specs/openapi.yml"
elif [ -f "drl-trading-ingest/specs/swagger.yaml" ]; then
    INGEST_SPEC="drl-trading-ingest/specs/swagger.yaml"
elif [ -f "drl-trading-ingest/specs/swagger.yml" ]; then
    INGEST_SPEC="drl-trading-ingest/specs/swagger.yml"
else
    echo "Error: Could not find OpenAPI spec in drl-trading-ingest/specs/"
    exit 1
fi

echo "Found spec: $INGEST_SPEC"

# Generate clients for services that need them
SERVICES=("drl-trading-training")

for service in "${SERVICES[@]}"; do
    if [ -d "$service" ]; then
        echo "Generating client for $service..."
        cd "$service"

        # Run the generate_clients.py script
        if [ -f "../scripts/generate_clients.py" ]; then
            python ../scripts/generate_clients.py
        else
            echo "Warning: generate_clients.py not found, skipping client generation for $service"
        fi

        cd ..
    fi
done

echo "Client generation completed."
