"""
Main entry point for DRL Trading Execution Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""

from drl_trading_execution.infrastructure.bootstrap.execution_service_bootstrap import bootstrap_execution_service


def main() -> None:
    """Main entry point for execution service."""
    bootstrap_execution_service()


if __name__ == "__main__":
    main()
