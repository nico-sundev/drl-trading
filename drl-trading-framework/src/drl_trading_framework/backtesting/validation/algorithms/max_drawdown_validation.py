from drl_trading_framework.backtesting.registry import register_validation
from drl_trading_framework.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_framework.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_framework.backtesting.validation.algorithms.config.max_drawdown_config import (
    MaxDrawdownConfig,
)
from drl_trading_framework.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class MaxDrawdownValidation(BaseValidationAlgorithm[MaxDrawdownConfig]):
    def __init__(self, config: MaxDrawdownConfig):
        super().__init__(config)
        self.name = "MaxDrawdownValidation"

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        equity = strategy.get_equity_curve()
        peak = equity.cummax()
        drawdowns = (equity - peak) / peak
        max_drawdown = drawdowns.min() * 100  # convert to percentage
        passed = abs(max_drawdown) <= self.config.max_drawdown_pct
        explanation = f"Max drawdown = {max_drawdown:.2f}%, limit = {self.config.max_drawdown_pct}%"

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=max_drawdown,
            threshold=self.config.max_drawdown_pct,
            explanation=explanation,
        )


register_validation("MaxDrawdownValidation", MaxDrawdownValidation, MaxDrawdownConfig)
