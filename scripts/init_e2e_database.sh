#!/bin/bash
# E2E Database Initialization Script
#
# This script ensures the database schema is properly initialized for E2E tests
# by running the drl-trading-ingest migrations. Since the ingest service owns
# the market_data table schema, we need to run its migrations before any E2E
# tests that depend on market data.
#
# Usage:
#   ./scripts/init_e2e_database.sh
#
# Prerequisites:
#   - TimescaleDB container running (via docker-compose)
#   - drl-trading-ingest dependencies installed

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INGEST_DIR="$PROJECT_ROOT/drl-trading-ingest"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== E2E Database Initialization ===${NC}"

# Check if TimescaleDB is accessible
echo -e "\n${YELLOW}[1/3] Checking database connectivity...${NC}"

# Export environment variables for migration (matching docker-compose setup)
export DB_HOST=${DB_HOST:-localhost}
export DB_PORT=${DB_PORT:-5432}
export DB_NAME=${DB_NAME:-marketdata}
export DB_USER=${DB_USER:-postgres}
export DB_PASSWORD=${DB_PASSWORD:-postgres}

# Test database connection with timeout
echo "Testing connection to postgres://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\q' 2>/dev/null; then
        echo -e "${GREEN}✓ Database is accessible${NC}"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}ERROR: Cannot connect to database after $max_attempts attempts${NC}"
        echo "Please ensure TimescaleDB is running:"
        echo "  docker-compose -f docker_compose/docker-compose.yml up -d timescaledb"
        exit 1
    fi

    echo "Waiting for database... (attempt $attempt/$max_attempts)"
    sleep 1
done

# Step 2: Run ingest service migrations
echo -e "\n${YELLOW}[2/3] Running drl-trading-ingest migrations...${NC}"

cd "$INGEST_DIR"

# Check if alembic is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}ERROR: uv is not installed${NC}"
    echo "Please install uv: pip install uv"
    exit 1
fi

# Set configuration for CI environment
export STAGE=ci
export SERVICE_CONFIG_PATH="$INGEST_DIR/config/application-cicd.yaml"

# Run migrations using the ingest migration CLI
echo "Applying migrations..."
uv run python -m drl_trading_ingest.adapter.cli.migration_cli migrate || {
    echo -e "${RED}ERROR: Migration failed${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check database logs: docker-compose -f docker_compose/docker-compose.yml logs timescaledb"
    echo "2. Verify environment variables:"
    echo "   DB_HOST=$DB_HOST"
    echo "   DB_PORT=$DB_PORT"
    echo "   DB_NAME=$DB_NAME"
    echo "   DB_USER=$DB_USER"
    echo "3. Check migration status: cd drl-trading-ingest && uv run python -m drl_trading_ingest.adapter.cli.migration_cli status"
    exit 1
}

echo -e "${GREEN}✓ Migrations applied successfully${NC}"

# Step 3: Verify schema
echo -e "\n${YELLOW}[3/3] Verifying schema...${NC}"

# Check if market_data table exists
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
    SELECT
        table_name,
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_name = 'market_data'
    ORDER BY ordinal_position;
" || {
    echo -e "${RED}ERROR: market_data table verification failed${NC}"
    exit 1
}

echo -e "\n${GREEN}✅ E2E database initialization complete${NC}"
echo ""
echo "The following tables are ready:"
echo "  - market_data (TimescaleDB hypertable)"
echo ""
echo "You can now run E2E tests:"
echo "  pytest tests/e2e/ -v"
