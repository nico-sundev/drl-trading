"""
Main entry point for DRL Trading Inference Service.

HEXAGONAL ARCHITECTURE:
- Minimal main.py (just infrastructure bootstrap)
- Business logic lives in core layer
- External interfaces live in adapter layer
"""
from drl_trading_inference.infrastructure.bootstrap.inference_service_bootstrap import bootstrap_inference_service


def main() -> None:
    """
    Main entry point for inference service.

    Uses standardized bootstrap pattern while maintaining
    hexagonal architecture compliance.
    """
    bootstrap_inference_service()


if __name__ == "__main__":
    main()
