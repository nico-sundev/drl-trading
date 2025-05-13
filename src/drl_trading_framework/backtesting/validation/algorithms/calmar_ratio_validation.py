from drl_trading_framework.backtesting.registry import register_validation
from drl_trading_framework.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_framework.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_framework.backtesting.validation.algorithms.config.calmar_ratio_config import (
    CalmarRatioConfig,
)
from drl_trading_framework.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class CalmarRatioValidation(BaseValidationAlgorithm[CalmarRatioConfig]):
    """
    Validates a trading strategy based on its Calmar ratio.

    The Calmar ratio is a performance measurement used to evaluate the risk-adjusted
    performance of investment funds, hedge funds, or trading strategies. It's calculated
    as the ratio between the compound annual growth rate and the maximum drawdown over
    a specified time period.

    A higher Calmar ratio indicates better risk-adjusted performance.
    """

    def __init__(self, config: CalmarRatioConfig):
        super().__init__(config)
        self.name = "CalmarRatioValidation"

        if self.config.min_calmar <= 0:
            raise ValueError("min_calmar must be positive")

        if self.config.lookback_years <= 0:
            raise ValueError("lookback_years must be positive")

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Calculate the Calmar ratio and validate against the minimum threshold.

        Args:
            strategy: The trading strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        equity = strategy.get_equity_curve()

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Calculate lookback period in days
        lookback_period = int(
            self.config.lookback_years * self.config.annualization_factor
        )

        # Use the full dataset if lookback period is longer than available data
        if lookback_period < len(returns):
            returns = returns.iloc[-lookback_period:]

        # Calculate annualized return
        compound_return = (1 + returns).prod() - 1
        years = len(returns) / self.config.annualization_factor
        annualized_return = (1 + compound_return) ** (1 / years) - 1

        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1) * 100
        max_drawdown = abs(drawdown.min()) / 100  # Convert to decimal

        # Calculate Calmar ratio (avoid division by zero)
        calmar_ratio = 0.0
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown

        # Determine if the strategy passes validation
        passed = calmar_ratio >= self.config.min_calmar

        # Use custom benchmark as threshold if provided
        threshold = (
            self.config.custom_benchmark
            if self.config.custom_benchmark is not None
            else self.config.min_calmar
        )

        explanation = (
            f"Calmar ratio = {calmar_ratio:.2f}, "
            f"threshold = {threshold:.2f}, "
            f"annualized return = {annualized_return:.2%}, "
            f"max drawdown = {max_drawdown:.2%}"
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=calmar_ratio,
            threshold=threshold,
            explanation=explanation,
        )


# Register the validation algorithm
register_validation("CalmarRatioValidation", CalmarRatioValidation, CalmarRatioConfig)
