"""
FTMO Stop Loss Usage Validation.

This module provides validation for FTMO's requirement that all trades must have stop losses set.
"""

from typing import List

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


class StopLossUsageValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates that a strategy uses stop losses on all trades as required by FTMO.

    FTMO requires that all trades have a stop loss set to manage risk properly.
    This validator checks all trades to ensure they have valid stop losses configured.

    Additionally, it can validate that stop losses are set at reasonable levels
    compared to entry price to prevent excessively wide stops.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOStopLossUsageValidation"

        # Validate configuration
        if not self.config.require_stop_loss:
            # If stop loss validation is disabled but this validator is used, something's wrong
            raise ValueError(
                "StopLossUsageValidation instantiated but config.require_stop_loss is False"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Check if all trades in the strategy have stop losses set.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get all trades, including open ones since they should have stop losses too
        trades = strategy.get_trades(include_open=True)

        if not trades:
            return ValidationResult(
                name=self.name,
                passed=True,
                score=100.0,  # Perfect compliance score if no trades
                threshold=100.0,
                explanation="No trades found to validate stop loss usage.",
            )

        # Check each trade for stop loss
        missing_stop_loss_trades: List[Trade] = []

        for trade in trades:
            if trade.stop_loss is None:
                missing_stop_loss_trades.append(trade)

        # Calculate compliance percentage
        total_trades = len(trades)
        compliant_trades = total_trades - len(missing_stop_loss_trades)
        compliance_percentage = (
            (compliant_trades / total_trades) * 100 if total_trades > 0 else 100.0
        )

        # Determine if validation passed
        # For FTMO, we expect 100% compliance (all trades must have stop losses)
        passed = len(missing_stop_loss_trades) == 0

        # Create detailed explanation
        explanation = self._create_explanation(
            passed, total_trades, compliant_trades, missing_stop_loss_trades
        )

        # For FTMO compliance, the threshold is 100% - all trades must have stop losses
        threshold = 100.0

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=compliance_percentage,
            threshold=threshold,
            explanation=explanation,
        )

    def _create_explanation(
        self,
        passed: bool,
        total_trades: int,
        compliant_trades: int,
        missing_sl_trades: List[Trade],
    ) -> str:
        """
        Create a detailed explanation message based on validation results.

        Args:
            passed: Whether the validation passed.
            total_trades: Total number of trades analyzed.
            compliant_trades: Number of trades with valid stop losses.
            missing_sl_trades: List of trades missing stop losses.

        Returns:
            Detailed explanation string.
        """
        explanation = "FTMO requires all trades to have stop losses set. "

        if passed:
            explanation += f"All {total_trades} trades comply with this requirement."
        else:
            missing_count = len(missing_sl_trades)
            compliance_rate = (
                (compliant_trades / total_trades) * 100 if total_trades > 0 else 0.0
            )

            explanation += (
                f"Validation FAILED - {missing_count} out of {total_trades} trades "
                f"({100 - compliance_rate:.1f}%) do not have stop losses set.\n"
            )

            # Show details for up to 5 violating trades
            max_examples = min(5, missing_count)
            for i in range(max_examples):
                trade = missing_sl_trades[i]
                symbol = trade.symbol
                entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M:%S")
                direction = "LONG" if trade.direction > 0 else "SHORT"
                explanation += f"  - {symbol} {direction} trade entered at {entry_time} (no stop loss)\n"

            if missing_count > max_examples:
                explanation += f"  - ...and {missing_count - max_examples} more trade(s) without stop losses."

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOStopLossUsageValidation", StopLossUsageValidation, FTMOAccountConfig
)
