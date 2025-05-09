from ai_trading.backtesting.registry import register_validation
from ai_trading.backtesting.strategy.strategy_interface import StrategyInterface
from ai_trading.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from ai_trading.backtesting.validation.algorithms.config.profit_factor_config import (
    ProfitFactorConfig,
)
from ai_trading.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class ProfitFactorValidation(BaseValidationAlgorithm[ProfitFactorConfig]):
    """
    Validates a trading strategy based on its Profit Factor.

    Profit Factor is defined as the gross profit divided by the gross loss.
    It's a measure of the profitability of a trading system relative to its
    losses. A profit factor greater than 1.0 indicates a profitable system.

    This validator checks if the strategy's profit factor meets or exceeds
    the minimum specified in the configuration.
    """

    def __init__(self, config: ProfitFactorConfig) -> None:
        super().__init__(config)
        self.name = "ProfitFactorValidation"

        if self.config.min_profit_factor < 1.0:
            raise ValueError(
                f"min_profit_factor should generally be >= 1.0 (profitable system), got {self.config.min_profit_factor}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Calculate the Profit Factor and validate against the minimum threshold.

        Args:
            strategy: The trading strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get trade history from the strategy
        trades = strategy.get_trades(include_open=self.config.include_open_trades)

        if not trades or len(trades) == 0:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.config.min_profit_factor,
                explanation="No trades found to calculate profit factor",
            )

        # Separate winning and losing trades
        winning_trades = [trade for trade in trades if trade.profit > 0]
        losing_trades = [trade for trade in trades if trade.profit < 0]

        # Calculate gross profits and gross losses
        gross_profit = sum(trade.profit for trade in winning_trades)
        gross_loss = abs(
            sum(trade.profit for trade in losing_trades)
        )  # Make positive for the ratio

        # Calculate profit factor (avoid division by zero)
        profit_factor = 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")  # No losses but some profits

        # Determine if the strategy passes validation
        threshold = (
            self.config.custom_benchmark
            if self.config.custom_benchmark is not None
            else self.config.min_profit_factor
        )
        passed = profit_factor >= threshold

        # Create explanation with detailed statistics
        explanation = (
            f"Profit Factor = {profit_factor:.2f}, threshold = {threshold:.2f}, "
            f"gross profit = {gross_profit:.2f}, gross loss = {gross_loss:.2f}, "
            f"winning trades = {len(winning_trades)}, losing trades = {len(losing_trades)}"
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=profit_factor,
            threshold=threshold,
            explanation=explanation,
        )


# Register the validation algorithm
register_validation(
    "ProfitFactorValidation", ProfitFactorValidation, ProfitFactorConfig
)
