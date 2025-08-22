from typing import Tuple

import pandas as pd

from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.max_consecutive_losses_config import (
    MaxConsecutiveLossesConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class MaxConsecutiveLossesValidation(
    BaseValidationAlgorithm[MaxConsecutiveLossesConfig]
):
    """
    Validates a trading strategy based on its maximum consecutive losing trades.

    This validator checks if a strategy exceeds a maximum allowed streak of consecutive
    losing trades, which is an important risk management measure. Optionally, it can also
    evaluate the drawdown during losing streaks.
    """

    def __init__(self, config: MaxConsecutiveLossesConfig):
        super().__init__(config)
        self.name = "MaxConsecutiveLossesValidation"

        if self.config.max_consecutive_losses < 1:
            raise ValueError(
                f"max_consecutive_losses must be positive, got {self.config.max_consecutive_losses}"
            )

        if (
            self.config.consider_drawdown
            and self.config.max_drawdown_during_streak <= 0
        ):
            raise ValueError(
                f"max_drawdown_during_streak must be positive, got {self.config.max_drawdown_during_streak}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Calculate the maximum consecutive losses and validate against the threshold.

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
                passed=True,
                score=0,
                threshold=self.config.max_consecutive_losses,
                explanation="No trades found to calculate consecutive losses",
            )

        # Create a DataFrame from trades for easier analysis
        trade_df = pd.DataFrame(
            [
                {
                    "timestamp": trade.exit_time,
                    "profit": trade.profit,
                    "is_loss": trade.profit < 0,
                    "equity": trade.cumulative_equity,
                }
                for trade in trades
                if trade.exit_time is not None
            ]
        )

        if trade_df.empty:
            return ValidationResult(
                name=self.name,
                passed=True,
                score=0,
                threshold=self.config.max_consecutive_losses,
                explanation="No closed trades found to calculate consecutive losses",
            )

        # Sort by timestamp to ensure chronological order
        trade_df = trade_df.sort_values("timestamp")

        # Find the maximum losing streak
        max_streak, streak_details = self._calculate_max_streak(trade_df)

        # Determine if the strategy passes validation
        passed = max_streak <= self.config.max_consecutive_losses

        # Add additional drawdown check if configured
        drawdown_explanation = ""
        if (
            self.config.consider_drawdown
            and streak_details
            and "max_drawdown" in streak_details
        ):
            max_drawdown = streak_details["max_drawdown"]
            drawdown_passed = max_drawdown <= self.config.max_drawdown_during_streak
            passed = passed and drawdown_passed
            drawdown_explanation = f", max drawdown during streak = {max_drawdown:.2f}% (limit {self.config.max_drawdown_during_streak:.2f}%)"

        explanation = f"Maximum consecutive losses = {max_streak} (limit {self.config.max_consecutive_losses}){drawdown_explanation}"

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=max_streak,
            threshold=self.config.max_consecutive_losses,
            explanation=explanation,
        )

    def _calculate_max_streak(self, trade_df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Calculate the maximum streak of consecutive losing trades.

        Args:
            trade_df: DataFrame containing trade data with 'is_loss' column.

        Returns:
            Tuple containing maximum streak count and additional details dictionary.
        """
        current_streak = 0
        max_streak = 0
        streak_details = {}

        # For drawdown calculation during streaks
        streak_start_equity = None
        min_equity_during_streak = None

        for _i, row in trade_df.iterrows():
            if row["is_loss"]:
                # Start or continue a losing streak
                if current_streak == 0:
                    # Start of a new losing streak
                    streak_start_equity = row["equity"]
                    min_equity_during_streak = row["equity"]
                else:
                    # Update minimum equity during streak
                    if row["equity"] < min_equity_during_streak:
                        min_equity_during_streak = row["equity"]

                current_streak += 1

                # Update max streak if current streak is larger
                if current_streak > max_streak:
                    max_streak = current_streak

                    # Calculate drawdown during streak if needed
                    if (
                        self.config.consider_drawdown
                        and streak_start_equity
                        and min_equity_during_streak
                    ):
                        max_drawdown = (
                            (streak_start_equity - min_equity_during_streak)
                            / streak_start_equity
                            * 100
                        )
                        streak_details["max_drawdown"] = max_drawdown
            else:
                # End of a losing streak
                current_streak = 0
                streak_start_equity = None
                min_equity_during_streak = None

        return max_streak, streak_details


# Register the validation algorithm
register_validation(
    "MaxConsecutiveLossesValidation",
    MaxConsecutiveLossesValidation,
    MaxConsecutiveLossesConfig,
)
