#!/bin/bash
# Helper script to run services with docker-compose profiles
#
# Usage:
#   ./run-services.sh infra              # Only infrastructure (Kafka, DBs)
#   ./run-services.sh preprocess         # Infrastructure + preprocess service
#   ./run-services.sh preprocess -d      # Infrastructure + preprocess service (daemon mode, no logs)
#   ./run-services.sh ingest             # Infrastructure + ingest service
#   ./run-services.sh all                # Everything
#   ./run-services.sh down               # Stop all services
#   ./run-services.sh restart <profile>  # Restart specific profile
#   ./run-services.sh logs <service>     # Tail logs for specific service

set -e

COMPOSE_FILE="docker-compose.yml"
COMPOSE_PROJECT="ai-trading"

# Check if daemon mode is requested
DAEMON_MODE=false
if [[ "$2" == "-d" ]] || [[ "$2" == "--daemon" ]]; then
    DAEMON_MODE=true
fi

# Change to script directory
cd "$(dirname "$0")"

case "${1:-infra}" in
    infra)
        echo "Starting infrastructure only (Kafka, TimescaleDB, Postgres)..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" down 2>/dev/null || true
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" up -d
        if [ "$DAEMON_MODE" = false ]; then
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" logs -f
        fi
        ;;
    preprocess)
        echo "Starting infrastructure + preprocess service..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile preprocess down 2>/dev/null || true
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile preprocess up -d --build
        if [ "$DAEMON_MODE" = false ]; then
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile preprocess logs -f
        fi
        ;;
    ingest)
        echo "Starting infrastructure + ingest service..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile ingest down 2>/dev/null || true
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile ingest up -d --build
        if [ "$DAEMON_MODE" = false ]; then
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile ingest logs -f
        fi
        ;;
    all)
        echo "Starting all services..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile all down 2>/dev/null || true
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile all up -d --build
        if [ "$DAEMON_MODE" = false ]; then
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile all logs -f
        fi
        ;;
    restart)
        if [ -z "$2" ]; then
            echo "Usage: $0 restart <profile>"
            exit 1
        fi
        echo "Restarting $2..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile "$2" down
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile "$2" up -d --build
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile "$2" logs -f
        ;;
    logs)
        if [ -z "$2" ]; then
            echo "Showing all logs..."
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" logs -f
        else
            echo "Showing logs for $2..."
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" logs -f "$2"
        fi
        ;;
    ps)
        echo "Showing running services..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" ps
        ;;
    down)
        echo "Stopping all services..."
        docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile all down
        ;;
    clean)
        echo "Stopping all services and removing volumes..."
        read -p "This will delete all data. Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" --profile all down -v
        fi
        ;;
    *)
        echo "Usage: $0 {infra|preprocess|ingest|all|down|restart|logs|ps|clean}"
        echo ""
        echo "Commands:"
        echo "  infra              - Start infrastructure only"
        echo "  preprocess         - Start infrastructure + preprocess service"
        echo "  ingest             - Start infrastructure + ingest service"
        echo "  all                - Start all services"
        echo "  restart <profile>  - Restart specific profile"
        echo "  logs [service]     - Show logs (optional: for specific service)"
        echo "  ps                 - Show running services"
        echo "  down               - Stop all services"
        echo "  clean              - Stop all and remove volumes (data loss!)"
        exit 1
        ;;
esac
