"""
Main entry point for DRL Trading Training Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""

from drl_trading_training.infrastructure.bootstrap.training_service_bootstrap import bootstrap_training_service


def main() -> None:
    """
    Main entry point for training service.

    Uses standardized bootstrap pattern while maintaining
    hexagonal architecture compliance.
    """
    bootstrap_training_service()


if __name__ == "__main__":
    main()
