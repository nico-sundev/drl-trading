"""
Main entry point for DRL Trading Preprocess Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_preprocess.infrastructure.bootstrap.preprocess_bootstrap import bootstrap_preprocess_service


def main() -> None:
    """Main entry point for preprocess service."""
    bootstrap_preprocess_service()


if __name__ == "__main__":
    main()
