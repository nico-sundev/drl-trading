"""
FTMO Trading Days Validation.

This module provides validation for FTMO's requirement that a trader meets
a minimum number of trading days during the challenge/verification phases.
"""

from datetime import date
from typing import List, Set

import pytz

from ai_trading.backtesting.registry import register_validation
from ai_trading.backtesting.strategy.strategy_interface import StrategyInterface
from ai_trading.backtesting.strategy.trade import Trade
from ai_trading.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from ai_trading.backtesting.validation.algorithms.config.ftmo_account_config import (
    FTMOAccountConfig,
)
from ai_trading.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class TradingDaysValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates that a strategy meets FTMO's minimum trading days requirement.

    FTMO requires that traders be active for a minimum number of days (typically 10)
    during the challenge and verification phases. This helps prevent traders from
    trying to hit profit targets with risky trades in just a few days.

    A "trading day" is defined as a day where at least one trade is opened or closed.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOTradingDaysValidation"

        if self.config.min_trading_days < 1:
            raise ValueError(
                f"min_trading_days must be positive, got {self.config.min_trading_days}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Check if the strategy meets the minimum trading days requirement.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get all trades, including open ones which also count for trading activity
        trades = strategy.get_trades(include_open=True)

        if not trades:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0,
                threshold=self.config.min_trading_days,
                explanation=f"No trades found. FTMO requires at least {self.config.min_trading_days} trading days.",
            )

        # Calculate trading days
        trading_days = self._calculate_trading_days(trades)
        trading_day_count = len(trading_days)

        # Determine if validation passed
        passed = trading_day_count >= self.config.min_trading_days

        # Create detailed explanation
        explanation = self._create_explanation(passed, trading_day_count, trading_days)

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=trading_day_count,
            threshold=self.config.min_trading_days,
            explanation=explanation,
        )

    def _calculate_trading_days(self, trades: List[Trade]) -> Set[date]:
        """
        Calculate the set of days with trading activity.

        A day is considered a trading day if at least one trade was opened or closed on that day.

        Args:
            trades: List of trades to analyze.

        Returns:
            Set of dates that had trading activity.
        """
        timezone = pytz.timezone(self.config.timezone)
        trading_days: Set[date] = set()

        for trade in trades:
            # Add entry day
            if trade.entry_time:
                entry_date = trade.entry_time.astimezone(timezone).date()
                trading_days.add(entry_date)

            # Add exit day if trade is closed
            if trade.exit_time:
                exit_date = trade.exit_time.astimezone(timezone).date()
                trading_days.add(exit_date)

        return trading_days

    def _create_explanation(
        self, passed: bool, trading_day_count: int, trading_days: Set[date]
    ) -> str:
        """
        Create a detailed explanation message based on trading days analysis.

        Args:
            passed: Whether the validation passed.
            trading_day_count: Number of days with trading activity.
            trading_days: Set of dates with trading activity.

        Returns:
            Detailed explanation string.
        """
        explanation = (
            f"FTMO requires at least {self.config.min_trading_days} trading days. "
        )

        if passed:
            explanation += f"Strategy has {trading_day_count} trading days, which meets the requirement."
        else:
            days_short = self.config.min_trading_days - trading_day_count
            explanation += (
                f"Validation FAILED - Strategy only has {trading_day_count} trading days, "
                f"which is {days_short} days short of the requirement."
            )

        # Add information about trading period
        if trading_days:
            sorted_days = sorted(trading_days)
            start_date = sorted_days[0]
            end_date = sorted_days[-1]
            total_period_days = (end_date - start_date).days + 1
            trading_frequency = (trading_day_count / total_period_days) * 100

            explanation += (
                f"\nTrading period: {start_date.isoformat()} to {end_date.isoformat()} "
                f"({total_period_days} calendar days, {trading_day_count} trading days, "
                f"{trading_frequency:.1f}% trading frequency)"
            )

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOTradingDaysValidation", TradingDaysValidation, FTMOAccountConfig
)
