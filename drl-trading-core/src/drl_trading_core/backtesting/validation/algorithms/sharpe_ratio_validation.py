from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.sharpe_ratio_config import (
    SharpeRatioConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class SharpeRatioValidation(BaseValidationAlgorithm[SharpeRatioConfig]):
    def __init__(self, config: SharpeRatioConfig):
        super().__init__(config)
        self.name = "SharpeRatioValidation"

        if not hasattr(self.config, "window"):
            raise ValueError("Config is missing 'window' attribute.")

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        equity = strategy.get_equity_curve()
        returns = equity.pct_change().dropna()
        mean = returns.mean()
        std = returns.std()
        sharpe = (mean / std) * (self.config.window**0.5)

        passed = sharpe >= self.config.min_sharpe
        explanation = (
            f"Sharpe ratio = {sharpe:.2f}, threshold = {self.config.min_sharpe}"
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=sharpe,
            threshold=self.config.min_sharpe,
            explanation=explanation,
        )


register_validation("SharpeRatioValidation", SharpeRatioValidation, SharpeRatioConfig)
