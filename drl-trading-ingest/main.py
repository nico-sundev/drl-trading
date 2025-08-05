"""
Main entry point for DRL Trading Ingest Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_ingest.infrastructure.bootstrap.ingest_function_bootstrap import bootstrap_ingest_service


def main() -> int:
    """
    Main entry point for the ingest service.

    Uses the standardized bootstrap pattern to start the service.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Initialize and run the bootstrap
        bootstrap_ingest_service()

    except KeyboardInterrupt:
        print("Service stopped by user")
        return 0
    except Exception as e:
        print(f"Failed to start ingest service: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
