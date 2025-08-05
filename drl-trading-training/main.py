
"""
Main entry point for DRL Trading Training Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_training.infrastructure.bootstrap.training_service_bootstrap import main


if __name__ == "__main__":
    main()
