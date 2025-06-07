from drl_trading_common.config.logging_config import configure_logging

from .training_app import TrainingApp

configure_logging()

__all__ = [
    "TrainingApp"
]
