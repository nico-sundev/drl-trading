import logging
import sys
from pathlib import Path


def configure_logging() -> None:
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "drl_trading_framework.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Configure Feast logger
    feast_logger = logging.getLogger("feast")
    feast_logger.setLevel(logging.INFO)

    # Configure feature store logger
    feature_store_logger = logging.getLogger("drl_trading_framework.feature_repo")
    feature_store_logger.setLevel(logging.INFO)
