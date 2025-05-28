#!/usr/bin/env python3
"""Main bootstrap script that adapts to deployment mode."""
# Added another comment for testing pre-commit hooks workflow

import os
import sys
from pathlib import Path

# Add framework to path
framework_path = Path(__file__).parent / "drl-trading-framework" / "src"
sys.path.insert(0, str(framework_path))


def main():
    """Main entry point that adapts to deployment mode."""

    # Detect deployment mode
    mode = os.getenv("DEPLOYMENT_MODE", "training").lower()
    service_type = os.getenv("SERVICE_TYPE", "training").lower()

    print(f"🚀 Starting in {mode} mode...")

    if mode == "training":
        run_training()
    elif mode == "production":
        run_production_service(service_type)
    else:
        raise ValueError(f"Unknown deployment mode: {mode}")


def run_training():
    """Run training mode - single process, all components."""
    print("🎓 Starting training mode...")

    # Import and run your existing training bootstrap
    from drl_trading_impl.main import run_training

    run_training()


def run_production_service(service_type: str):
    """Run specific production service."""
    print(f"🏭 Starting production service: {service_type}")

    if service_type == "data_ingestion":
        from scripts.production.data_ingestion_service import main as run_data_ingestion

        run_data_ingestion()

    elif service_type == "inference":
        symbol = os.getenv("TRADING_SYMBOL", "EURUSD")
        from scripts.production.inference_service import main as run_inference

        run_inference(symbol)

    elif service_type == "execution":
        from scripts.production.execution_service import main as run_execution

        run_execution()

    else:
        raise ValueError(f"Unknown service type: {service_type}")


if __name__ == "__main__":
    main()
