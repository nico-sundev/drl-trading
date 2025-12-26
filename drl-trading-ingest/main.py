"""
Main entry point for DRL Trading Ingest Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_ingest.application.bootstrap.ingest_service_bootstrap import bootstrap_ingest_service


def main() -> None:
    """
    Main entry point for the ingest service.

    Uses the standardized bootstrap pattern to start the service.
    """
    bootstrap_ingest_service()


if __name__ == "__main__":
    main()
