import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dask import compute, delayed
from rich.console import Console
from rich.table import Table

from drl_trading_core.backtesting.registry import VALIDATION_REGISTRY
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.container.backtest_validation_result import (
    BacktestValidationSummary,
)
from drl_trading_core.backtesting.validation.container.overall_status import (
    OverallStatus,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)

# --- Logging Setup ---
logger = logging.getLogger("backtest.validation")
logger.setLevel(logging.INFO)
console = Console()


# --- Validator Interface ---
class BacktestValidatorInterface(ABC):
    """Interface defining contract for backtest validation functionality."""

    @abstractmethod
    def validate(
        self, strategy: StrategyInterface, algorithms: List[Dict[str, Any]]
    ) -> BacktestValidationSummary:
        """
        Validate a strategy using the provided validation algorithms.

        Args:
            strategy: The trading strategy to validate
            algorithms: List of validation algorithms to apply

        Returns:
            A summary of validation results
        """
        pass


# --- Validator Implementation ---
class BacktestValidator(BacktestValidatorInterface):
    """
    Validates trading strategies using configurable validation algorithms.

    This class implements a stateless approach to backtest validation, where
    both the strategy and algorithms are provided at validation time rather
    than stored as instance state.
    """

    def validate(
        self, strategy: StrategyInterface, validators: List[Dict[str, Any]]
    ) -> BacktestValidationSummary:
        """
        Run validation algorithms against the strategy and produce a summary.

        Args:
            strategy: The trading strategy to validate
            algorithms: List of validation algorithms to apply

        Returns:
            A detailed summary of all validation results
        """
        algorithms: list[BaseValidationAlgorithm] = self._parse_validations(validators)
        tasks = [delayed(alg.run)(strategy) for alg in algorithms]
        results: List[ValidationResult] = compute(*tasks, scheduler="threads")

        failed = [r.name for r in results if not r.passed]
        overall_status = OverallStatus.PASS if not failed else OverallStatus.FAIL

        summary = BacktestValidationSummary.from_results(
            results=results,
            overall_status=overall_status,
            config_snapshot={
                alg.name: getattr(alg, "config", {}) for alg in algorithms
            },
        )

        self._log_summary(summary)
        return summary

    def _parse_validations(
        self, validations: List[Dict[str, Any]]
    ) -> List[BaseValidationAlgorithm]:
        """
        Parse and instantiate validation algorithms from a list of configurations.

        Args:
            validations (List[Dict[str, Any]]): List of validation configurations.

        Raises:
            ValueError: If a validation algorithm is not registered.

        Returns:
            List[BaseValidationAlgorithm]: List of instantiated validation algorithms.
        """
        algorithms: List[BaseValidationAlgorithm] = []

        for entry in validations:
            name = entry["name"]
            config_kwargs = entry.get("config", {})

            if name not in VALIDATION_REGISTRY:
                raise ValueError(f"Validation algorithm '{name}' is not registered.")

            alg_cls, config_cls = VALIDATION_REGISTRY[name]
            config = config_cls(**config_kwargs)
            algorithm_instance = alg_cls(config)
            algorithms.append(algorithm_instance)

        return algorithms

    def _log_summary(self, summary: BacktestValidationSummary) -> None:
        """
        Log validation results in a formatted table.

        Args:
            summary: The validation summary to log
        """
        table = Table(title="Backtest Validation Results")
        table.add_column("Check")
        table.add_column("Passed")
        table.add_column("Score")
        table.add_column("Threshold")
        table.add_column("Explanation")

        for result in summary.results:
            table.add_row(
                result.name,
                "✅" if result.passed else "❌",
                f"{result.score:.2f}" if result.score is not None else "-",
                str(result.threshold),
                result.explanation,
            )

        logger.info("Validation Summary:")
        console.print(table)
