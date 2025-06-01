from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Tuple

import pytz

from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.strategy.trade import Trade
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.ftmo_account_config import (
    FTMOAccountConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class MaxDailyLossValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates a trading strategy against FTMO's maximum daily loss rule.

    FTMO requires that at no point should the daily loss exceed a certain percentage
    of the account size (typically 5%). This validator checks all trading days to
    ensure this rule is never violated.

    Note: This validation uses the closing balance at the end of each trading day
    to determine compliance, not the intraday minimum balance.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOMaxDailyLossValidation"

        if (
            self.config.max_daily_loss_percent <= 0
            or self.config.max_daily_loss_percent >= 100
        ):
            raise ValueError(
                f"max_daily_loss_percent must be between 0 and 100, got {self.config.max_daily_loss_percent}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Check if the strategy violated the maximum daily loss rule on any day.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get all trades
        trades = strategy.get_trades(include_open=False)

        if not trades:
            return ValidationResult(
                name=self.name,
                passed=True,
                score=0.0,
                threshold=self.config.max_daily_loss_percent,
                explanation="No closed trades to analyze for daily loss compliance.",
            )

        # Calculate daily P&L and ending balances
        daily_results = self._calculate_daily_results(trades)

        # Calculate maximum allowed daily loss
        max_allowed_loss = self.config.account_size * (
            self.config.max_daily_loss_percent / 100
        )

        # Find any days that violated the rule
        violations = []
        worst_daily_loss = 0.0
        worst_daily_loss_pct = 0.0

        for day, (
            _starting_balance,
            _ending_balance,
            daily_pnl,
        ) in daily_results.items():
            # Skip days with no losses
            if daily_pnl >= 0:
                continue

            daily_loss = abs(daily_pnl)
            daily_loss_pct = (daily_loss / self.config.account_size) * 100

            if daily_loss > max_allowed_loss:
                violations.append((day, daily_loss, daily_loss_pct))

            # Track the worst daily loss for reporting
            if daily_loss > worst_daily_loss:
                worst_daily_loss = daily_loss
                worst_daily_loss_pct = daily_loss_pct

        # Determine if validation passed
        passed = len(violations) == 0

        # Create detailed explanation
        explanation = self._create_explanation(
            passed, violations, worst_daily_loss, worst_daily_loss_pct
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=worst_daily_loss_pct,
            threshold=self.config.max_daily_loss_percent,
            explanation=explanation,
        )

    def _calculate_daily_results(
        self, trades: List[Trade]
    ) -> Dict[date, Tuple[float, float, float]]:
        """
        Calculate starting balance, ending balance, and daily P&L for each trading day.

        Args:
            trades: List of completed trades.

        Returns:
            Dictionary mapping trading dates to (starting_balance, ending_balance, daily_pnl) tuples.
        """
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)

        # Group trades by day
        daily_trades = defaultdict(list)
        timezone = pytz.timezone(self.config.timezone)

        for trade in sorted_trades:
            # Skip trades without exit time
            if trade.exit_time is None:
                continue

            # Convert to the configured timezone and extract date
            trade_date = trade.exit_time.astimezone(timezone).date()
            daily_trades[trade_date].append(trade)

        # Calculate daily results
        daily_results = {}
        prior_day_balance = self.config.account_size  # Starting account balance

        for day in sorted(daily_trades.keys()):
            day_trades = daily_trades[day]

            # Starting balance for the day is the ending balance of the previous day
            starting_balance = prior_day_balance

            # Calculate total P&L for the day
            daily_pnl = sum(trade.profit for trade in day_trades)

            # Ending balance for the day
            ending_balance = starting_balance + daily_pnl

            # Store results
            daily_results[day] = (starting_balance, ending_balance, daily_pnl)

            # Update prior day balance for the next iteration
            prior_day_balance = ending_balance

        return daily_results

    def _create_explanation(
        self,
        passed: bool,
        violations: List[Tuple[date, float, float]],
        worst_daily_loss: float,
        worst_daily_loss_pct: float,
    ) -> str:
        """
        Create a detailed explanation message based on validation results.

        Args:
            passed: Whether the validation passed.
            violations: List of (date, loss_amount, loss_percent) tuples for days that violated the rule.
            worst_daily_loss: The largest daily loss amount.
            worst_daily_loss_pct: The largest daily loss as a percentage.

        Returns:
            Detailed explanation string.
        """
        explanation = f"FTMO maximum daily loss requirement: {self.config.max_daily_loss_percent:.2f}% ({self.config.max_daily_loss_percent/100*self.config.account_size:.2f} {self.config.currency}). "

        if passed:
            explanation += f"Worst daily loss was {worst_daily_loss_pct:.2f}% ({worst_daily_loss:.2f} {self.config.currency}), which is within limits."
        else:
            explanation += f"Validation FAILED - Daily loss limit exceeded on {len(violations)} day(s).\n"
            for day_date, loss_amount, loss_pct in violations[
                :3
            ]:  # Show max 3 violations
                explanation += f"  - {day_date.isoformat()}: Loss of {loss_pct:.2f}% ({loss_amount:.2f} {self.config.currency})\n"

            if len(violations) > 3:
                explanation += f"  - ...and {len(violations) - 3} more violation(s)."

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOMaxDailyLossValidation", MaxDailyLossValidation, FTMOAccountConfig
)
