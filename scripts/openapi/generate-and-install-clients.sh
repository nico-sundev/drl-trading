#!/bin/bash
# Generate and install OpenAPI clients for DRL Trading services

set -e

echo "Generating OpenAPI clients..."

# Services that need client generation
SERVICES=("drl-trading-training")

# Generate clients for each service
for service in "${SERVICES[@]}"; do
    if [ -d "$service" ]; then
        echo "Generating client for $service..."
        cd "$service"
        uv run generate-clients
        cd ..
        echo "Client generation completed for $service"
    else
        echo "Warning: Service directory not found: $service"
    fi
done

echo "All client generation completed."
