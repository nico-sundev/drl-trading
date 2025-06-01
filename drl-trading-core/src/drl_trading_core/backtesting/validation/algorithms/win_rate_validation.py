from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.win_rate_config import (
    WinRateConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class WinRateValidation(BaseValidationAlgorithm[WinRateConfig]):
    """
    Validates a trading strategy based on its win rate (percentage of profitable trades).

    This validator checks if a strategy maintains a minimum win rate, which is an
    important performance measure. It can also validate that the strategy is profitable
    overall in addition to meeting the win rate requirement.
    """

    def __init__(self, config: WinRateConfig) -> None:
        super().__init__(config)
        self.name = "WinRateValidation"

        if not 0 <= self.config.min_win_rate <= 100:
            raise ValueError(
                f"min_win_rate must be between 0 and 100, got {self.config.min_win_rate}"
            )

        if self.config.min_trade_count < 1:
            raise ValueError(
                f"min_trade_count must be positive, got {self.config.min_trade_count}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Calculate the win rate and validate against the minimum threshold.

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
                threshold=self.config.min_win_rate,
                explanation="No trades found to calculate win rate",
            )

        # Only consider closed trades for win rate calculation
        closed_trades = [trade for trade in trades if trade.exit_time is not None]

        if len(closed_trades) < self.config.min_trade_count:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.config.min_win_rate,
                explanation=f"Insufficient trades to calculate win rate: {len(closed_trades)} found, {self.config.min_trade_count} required",
            )

        # Calculate win rate
        winning_trades = [trade for trade in closed_trades if trade.profit > 0]
        total_trades = len(closed_trades)
        win_rate = (len(winning_trades) / total_trades) * 100

        # Check if the strategy is profitable overall if required
        total_profit = sum(trade.profit for trade in closed_trades)
        is_profitable = total_profit > 0

        # Determine if the strategy passes validation
        passed = win_rate >= self.config.min_win_rate
        if self.config.require_profit:
            passed = passed and is_profitable

        # Create explanation with detailed statistics
        profit_explanation = ""
        if self.config.require_profit:
            profit_explanation = f", net profit = {total_profit:.2f} ({'positive' if is_profitable else 'negative'})"

        explanation = (
            f"Win rate = {win_rate:.2f}%, threshold = {self.config.min_win_rate:.2f}%, "
            f"winning trades = {len(winning_trades)}, total trades = {total_trades}{profit_explanation}"
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=win_rate,
            threshold=self.config.min_win_rate,
            explanation=explanation,
        )


# Register the validation algorithm
register_validation("WinRateValidation", WinRateValidation, WinRateConfig)
