import logging
from typing import Any, Dict, List

from dask import compute, delayed
from rich.console import Console
from rich.table import Table

from ai_trading.backtesting.registry import VALIDATION_REGISTRY
from ai_trading.backtesting.strategy.strategy_interface import StrategyInterface
from ai_trading.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from ai_trading.backtesting.validation.container.backtest_validation_result import (
    BacktestValidationSummary,
)
from ai_trading.backtesting.validation.container.overall_status import OverallStatus
from ai_trading.backtesting.validation.container.validation_result import (
    ValidationResult,
)

# --- Logging Setup ---
logger = logging.getLogger("backtest.validation")
logger.setLevel(logging.INFO)
console = Console()


# --- Validator Entry Class ---
class BacktestValidator:
    def __init__(self, strategy: StrategyInterface) -> None:
        self.strategy = strategy
        self.algorithms: List[BaseValidationAlgorithm] = []

    @classmethod
    def from_registry(
        cls, strategy: StrategyInterface, validations: List[Dict[str, Any]]
    ) -> "BacktestValidator":
        instance = cls(strategy)
        for entry in validations:
            name = entry["name"]
            config_kwargs = entry.get("config", {})

            if name not in VALIDATION_REGISTRY:
                raise ValueError(f"Validation algorithm '{name}' is not registered.")

            alg_cls, config_cls = VALIDATION_REGISTRY[name]
            config = config_cls(**config_kwargs)
            algorithm_instance = alg_cls(config)  # Pass config directly to constructor
            instance.algorithms.append(algorithm_instance)

        return instance

    def validate(self) -> BacktestValidationSummary:
        tasks = [delayed(alg.run)(self.strategy) for alg in self.algorithms]
        results: List[ValidationResult] = compute(*tasks, scheduler="threads")

        failed = [r.name for r in results if not r.passed]
        overall_status = OverallStatus.PASS if not failed else OverallStatus.FAIL

        summary = BacktestValidationSummary.from_results(
            results=results,
            overall_status=overall_status,
            config_snapshot={
                alg.name: getattr(alg, "config", {}) for alg in self.algorithms
            },
        )

        self._log_summary(summary)
        return summary

    def _log_summary(self, summary: BacktestValidationSummary) -> None:
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
