#!/bin/bash
# E2E Test Runner for Preprocess Service
#
# This script automates the E2E testing workflow:
# 1. Start Docker Compose (infrastructure dependencies)
# 2. Start the preprocess service
# 3. Run E2E tests
# 4. Cleanup
#
# Usage:
#   ./scripts/run_e2e_tests.sh [--keep-running]
#
# Options:
#   --keep-running    Keep services running after tests (for debugging)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/drl-trading-preprocess"

KEEP_RUNNING=false
if [[ "$1" == "--keep-running" ]]; then
    KEEP_RUNNING=true
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== E2E Test Runner for Preprocess Service ===${NC}"

# Step 1: Start Docker Compose infrastructure
echo -e "\n${YELLOW}[1/5] Starting infrastructure (Docker Compose)...${NC}"
cd "$PROJECT_ROOT"
docker-compose -f docker_compose/docker-compose.yml up -d

# Wait for Kafka to be ready
echo -e "${YELLOW}[2/5] Waiting for Kafka to be ready...${NC}"
sleep 10

# Check Kafka health
echo "Checking Kafka connection..."
docker-compose -f docker_compose/docker-compose.yml exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --list || {
    echo -e "${RED}ERROR: Kafka is not responding${NC}"
    docker-compose -f docker_compose/docker-compose.yml logs kafka
    exit 1
}

echo -e "${GREEN}✓ Kafka is ready${NC}"

# Step 2: Start preprocess service
echo -e "\n${YELLOW}[3/5] Starting preprocess service...${NC}"
cd "$SERVICE_DIR"

# Kill any existing service process
pkill -f "drl-trading-preprocess" || true

# Start service in background
STAGE=ci python main.py > logs/e2e-test.log 2>&1 &
SERVICE_PID=$!

echo "Service PID: $SERVICE_PID"

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 5

# Check if service is still running
if ! ps -p $SERVICE_PID > /dev/null; then
    echo -e "${RED}ERROR: Service failed to start${NC}"
    cat logs/e2e-test.log
    exit 1
fi

echo -e "${GREEN}✓ Service is running${NC}"

# Step 3: Run E2E tests
echo -e "\n${YELLOW}[4/5] Running E2E tests...${NC}"
cd "$PROJECT_ROOT"

# Run pytest with E2E test markers
pytest tests/e2e/ \
    -v \
    --tb=short \
    --log-cli-level=INFO \
    -m "not skip" || TEST_FAILED=true

# Step 4: Cleanup (unless --keep-running)
if [[ "$KEEP_RUNNING" == true ]]; then
    echo -e "\n${YELLOW}Keeping services running for debugging${NC}"
    echo "Service PID: $SERVICE_PID"
    echo "View logs: tail -f $SERVICE_DIR/logs/e2e-test.log"
    echo ""
    echo "To stop services manually:"
    echo "  kill $SERVICE_PID"
    echo "  docker-compose -f docker_compose/docker-compose.yml down"
else
    echo -e "\n${YELLOW}[5/5] Cleaning up...${NC}"

    # Stop service
    echo "Stopping preprocess service (PID: $SERVICE_PID)..."
    kill $SERVICE_PID 2>/dev/null || true
    sleep 2

    # Stop Docker Compose
    echo "Stopping Docker Compose..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker_compose/docker-compose.yml down

    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

# Final result
if [[ "$TEST_FAILED" == true ]]; then
    echo -e "\n${RED}❌ E2E Tests FAILED${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ E2E Tests PASSED${NC}"
    exit 0
fi
