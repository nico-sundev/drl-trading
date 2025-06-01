"""
FTMO Maximum Position Size Validation.

This module provides validation for FTMO's maximum position size rule,
which limits how large individual positions can be to manage risk.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


class MaxPositionSizeValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates that a strategy respects FTMO's maximum position size limits.

    FTMO imposes limits on the maximum size of any individual position to manage risk.
    These limits typically scale with account size (e.g., 10 lots per $100K).

    This validator checks that no trade exceeds the maximum allowed position size,
    and provides detailed information about any violations.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOMaxPositionSizeValidation"

        if self.config.max_position_size is None:
            # Default calculation: 10 lots per $100K of account size
            self.max_position_size = self.config.account_size / 100000 * 10
        else:
            self.max_position_size = self.config.max_position_size

        if self.max_position_size <= 0:
            raise ValueError(
                f"max_position_size must be positive, got {self.max_position_size}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Check if the strategy respects the maximum position size limit.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get all trades, including open ones
        trades = strategy.get_trades(include_open=True)

        if not trades:
            return ValidationResult(
                name=self.name,
                passed=True,
                score=0.0,
                threshold=self.max_position_size,
                explanation=f"No trades found to validate against position size limit of {self.max_position_size} lots.",
            )

        # Check each trade against position size limit
        violations: List[Tuple[Optional[Trade], float]] = []
        max_position_found = 0.0

        for trade in trades:
            if trade.size > self.max_position_size:
                violations.append((trade, trade.size))

            max_position_found = max(max_position_found, trade.size)

        # Check cumulative positions per symbol at any given time
        symbol_positions = self._calculate_concurrent_positions(trades)
        for _symbol, positions in symbol_positions.items():
            for _timestamp, total_size in positions.items():
                if total_size > self.max_position_size:
                    violations.append(
                        (None, total_size)
                    )  # Use None to indicate a cumulative violation
                    max_position_found = max(max_position_found, total_size)

        # Determine if validation passed
        passed = len(violations) == 0

        # Create detailed explanation
        explanation = self._create_explanation(passed, max_position_found, violations)

        # For score, use the percentage of the limit used by the largest position
        score = (
            (max_position_found / self.max_position_size) * 100
            if self.max_position_size > 0
            else 0.0
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=score,
            threshold=100.0,  # Threshold is 100% of the limit
            explanation=explanation,
        )

    def _calculate_concurrent_positions(self, trades: List[Trade]) -> Dict[str, Dict]:
        """
        Calculate the total position size per symbol at each point in time.

        This method tracks the cumulative position size for each symbol to identify
        if multiple trades in the same symbol exceed the position size limit when combined.

        Args:
            trades: List of trades to analyze.

        Returns:
            Dictionary mapping symbols to timestamps and their total position sizes.
        """
        # Group trades by symbol
        symbol_trades = defaultdict(list)
        for trade in trades:
            symbol_trades[trade.symbol].append(trade)

        # For each symbol, calculate position sizes at each timestamp
        symbol_positions = {}

        for symbol, symbol_trade_list in symbol_trades.items():
            # Create a timeline of position changes
            position_changes = []

            for trade in symbol_trade_list:
                # Position increase at entry time
                position_changes.append(
                    (trade.entry_time, trade.size * trade.direction)
                )

                # Position decrease at exit time (if trade is closed)
                if trade.exit_time:
                    position_changes.append(
                        (trade.exit_time, -trade.size * trade.direction)
                    )

            # Sort changes by timestamp
            position_changes.sort(key=lambda x: x[0])

            # Calculate running position size
            current_position = 0.0
            positions_over_time = {}

            for timestamp, change in position_changes:
                current_position += change
                positions_over_time[timestamp] = abs(
                    current_position
                )  # Use absolute value for net position size

            symbol_positions[symbol] = positions_over_time

        return symbol_positions

    def _create_explanation(
        self,
        passed: bool,
        max_position_found: float,
        violations: List[Tuple[Optional[Trade], float]],
    ) -> str:
        """
        Create a detailed explanation message based on position size analysis.

        Args:
            passed: Whether the validation passed.
            max_position_found: Maximum position size found across all trades.
            violations: List of trades that violated the position size limit and their sizes.

        Returns:
            Detailed explanation string.
        """
        explanation = (
            f"FTMO maximum position size limit: {self.max_position_size} lots. "
        )

        if passed:
            explanation += f"Largest position was {max_position_found} lots, which is within the limit."
        else:
            explanation += (
                f"Validation FAILED - {len(violations)} position size violation(s) found. "
                f"Maximum allowed: {self.max_position_size} lots, largest found: {max_position_found} lots."
                f"\nViolations:"
            )

            # Show details for up to 5 violations
            individual_violations = [v for v in violations if v[0] is not None]
            cumulative_violations = [v for v in violations if v[0] is None]

            # Individual trade violations
            count = 0
            for trade, size in individual_violations[:3]:
                symbol = trade.symbol if trade is not None else "Unknown"
                entry_time = (
                    trade.entry_time.strftime("%Y-%m-%d %H:%M:%S")
                    if trade and trade.entry_time
                    else "Unknown"
                )
                direction = "LONG" if trade and trade.direction > 0 else "SHORT"
                explanation += f"\n  - Individual trade: {symbol} {direction} at {entry_time}, size: {size} lots"
                count += 1

            # Cumulative position violations
            for _, size in cumulative_violations[:2]:
                explanation += f"\n  - Cumulative position: size: {size} lots (combined positions in same symbol)"
                count += 1

            if len(violations) > count:
                explanation += (
                    f"\n  - ...and {len(violations) - count} more violation(s)."
                )

            # Add recommendation
            excessive_pct = (max_position_found / self.max_position_size - 1) * 100
            explanation += f"\n\nRecommendation: Reduce largest position sizes by at least {excessive_pct:.1f}% to comply with FTMO rules."

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOMaxPositionSizeValidation", MaxPositionSizeValidation, FTMOAccountConfig
)
