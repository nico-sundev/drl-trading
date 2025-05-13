"""
FTMO Maximum Drawdown Validation.

This module provides validation for FTMO's requirement that the maximum drawdown
never exceeds their specified limits (typically 10% of account size).
"""

from typing import Optional, Tuple

import pandas as pd

from drl_trading_framework.backtesting.registry import register_validation
from drl_trading_framework.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_framework.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_framework.backtesting.validation.algorithms.config.ftmo_account_config import (
    FTMOAccountConfig,
)
from drl_trading_framework.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class FTMOMaxDrawdownValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates a trading strategy against FTMO's maximum drawdown rule.

    FTMO requires that the maximum drawdown never exceeds a certain percentage
    of the account size (typically 10%). This validator calculates the maximum
    drawdown from the equity curve and ensures it stays within the allowed limits.

    The validator analyzes the entire equity curve to identify the maximum
    peak-to-trough decline in percentage terms.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOMaxDrawdownValidation"

        if (
            self.config.max_total_loss_percent <= 0
            or self.config.max_total_loss_percent >= 100
        ):
            raise ValueError(
                f"max_total_loss_percent must be between 0 and 100, got {self.config.max_total_loss_percent}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Calculate the maximum drawdown and validate against the FTMO threshold.

        Args:
            strategy: The trading strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get the equity curve
        equity = strategy.get_equity_curve()

        if equity.empty:
            return ValidationResult(
                name=self.name,
                passed=True,
                score=0.0,
                threshold=self.config.max_total_loss_percent,
                explanation="No equity data available to calculate drawdown.",
            )

        # Calculate maximum drawdown and when it occurred
        max_dd_pct, max_dd_amount, dd_start_date, dd_end_date, dd_recovery_date = (
            self._calculate_drawdown_metrics(equity)
        )

        # Determine if validation passed
        passed = max_dd_pct <= self.config.max_total_loss_percent

        # Create detailed explanation
        explanation = self._create_explanation(
            passed,
            max_dd_pct,
            max_dd_amount,
            dd_start_date,
            dd_end_date,
            dd_recovery_date,
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=max_dd_pct,
            threshold=self.config.max_total_loss_percent,
            explanation=explanation,
        )

    def _calculate_drawdown_metrics(
        self, equity: pd.Series
    ) -> Tuple[float, float, str, str, Optional[str]]:
        """
        Calculate comprehensive drawdown metrics from the equity curve.

        Args:
            equity: Time series of account equity values.

        Returns:
            Tuple containing:
            - Maximum drawdown percentage
            - Maximum drawdown amount
            - Start date of the worst drawdown period
            - End date of the worst drawdown period (low point)
            - Recovery date or None if not recovered
        """
        # Running maximum equity
        running_max = equity.cummax()

        # Calculate drawdown series in both percentage and absolute terms
        drawdown_pct = ((equity - running_max) / running_max) * 100
        drawdown_amount = equity - running_max

        # Find the worst drawdown
        worst_dd_idx = drawdown_pct.idxmin()
        max_dd_pct = abs(drawdown_pct.min())
        max_dd_amount = abs(drawdown_amount.min())

        # Find the start date of the drawdown (last peak before the trough)
        # This is the last time equity was at its maximum before the worst drawdown
        dd_start_mask = (running_max.shift(1) != running_max) & (
            running_max.shift(1).notna()
        )
        dd_start_dates = equity.index[dd_start_mask]
        dd_start_date = (
            dd_start_dates[dd_start_dates <= worst_dd_idx][-1]
            if not dd_start_dates.empty
            else equity.index[0]
        )

        # The end date is when drawdown was at its worst
        dd_end_date = worst_dd_idx

        # Find the recovery date (when equity returns to the previous peak)
        # This is the first time after the worst drawdown that equity reaches or exceeds the previous peak
        recovery_mask = (equity >= running_max[dd_start_date]) & (
            equity.index > worst_dd_idx
        )
        dd_recovery_date = (
            equity.index[recovery_mask][0] if any(recovery_mask) else None
        )

        # Format dates as strings
        dd_start_str = dd_start_date.strftime("%Y-%m-%d %H:%M:%S")
        dd_end_str = dd_end_date.strftime("%Y-%m-%d %H:%M:%S")
        dd_recovery_str = (
            dd_recovery_date.strftime("%Y-%m-%d %H:%M:%S")
            if dd_recovery_date is not None
            else None
        )

        return max_dd_pct, max_dd_amount, dd_start_str, dd_end_str, dd_recovery_str

    def _create_explanation(
        self,
        passed: bool,
        max_dd_pct: float,
        max_dd_amount: float,
        dd_start_date: str,
        dd_end_date: str,
        dd_recovery_date: Optional[str],
    ) -> str:
        """
        Create a detailed explanation message based on drawdown analysis.

        Args:
            passed: Whether the validation passed.
            max_dd_pct: Maximum drawdown percentage.
            max_dd_amount: Maximum drawdown amount.
            dd_start_date: Start date of the worst drawdown.
            dd_end_date: End date (low point) of the worst drawdown.
            dd_recovery_date: Recovery date or None if not recovered.

        Returns:
            Detailed explanation string.
        """
        explanation = (
            f"FTMO maximum drawdown requirement: {self.config.max_total_loss_percent:.2f}% "
            f"({self.config.max_total_loss_percent/100*self.config.account_size:.2f} {self.config.currency}). "
        )

        if passed:
            explanation += (
                f"Maximum drawdown was {max_dd_pct:.2f}% ({max_dd_amount:.2f} {self.config.currency}), "
                f"which is within acceptable limits."
            )
        else:
            explanation += (
                f"Validation FAILED - Maximum drawdown of {max_dd_pct:.2f}% "
                f"({max_dd_amount:.2f} {self.config.currency}) exceeds the "
                f"{self.config.max_total_loss_percent:.2f}% limit.\n"
            )

        # Add detailed information about the drawdown period
        explanation += f"\nDrawdown period: {dd_start_date} to {dd_end_date}"

        if dd_recovery_date:
            explanation += f"\nRecovery date: {dd_recovery_date}"
        else:
            explanation += "\nDrawdown not recovered by the end of the testing period."

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOMaxDrawdownValidation", FTMOMaxDrawdownValidation, FTMOAccountConfig
)
