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
            logging.FileHandler(log_dir / "drl_trading_common.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
