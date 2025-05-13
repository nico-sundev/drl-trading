# --- Base Validation Algorithm ---
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from drl_trading_framework.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_framework.backtesting.validation.container.validation_result import (
    ValidationResult,
)

# Define a type variable for config types
TConfig = TypeVar("TConfig")


class BaseValidationAlgorithm(Generic[TConfig], ABC):
    name: str

    def __init__(self, config: TConfig) -> None:
        """
        Initialize the validation algorithm with its configuration.

        Args:
            config: The configuration for this validation algorithm
        """
        self.config: TConfig = config

    @abstractmethod
    def run(self, strategy: StrategyInterface) -> ValidationResult:
        pass
