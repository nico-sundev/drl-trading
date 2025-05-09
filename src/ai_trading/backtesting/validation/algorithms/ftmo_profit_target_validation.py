"""
FTMO Profit Target Validation.

This module provides validation for FTMO's profit target requirement,
which traders must meet to successfully pass the challenge/verification phases.
"""

from datetime import datetime
from typing import Optional

from ai_trading.backtesting.registry import register_validation
from ai_trading.backtesting.strategy.strategy_interface import StrategyInterface
from ai_trading.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from ai_trading.backtesting.validation.algorithms.config.ftmo_account_config import (
    FTMOAccountConfig,
)
from ai_trading.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class ProfitTargetValidation(BaseValidationAlgorithm[FTMOAccountConfig]):
    """
    Validates that a strategy meets FTMO's profit target requirement.

    FTMO requires traders to reach a specific profit target (typically 10% for
    smaller accounts, 8% for larger accounts) to successfully pass the challenge
    and verification phases.

    This validator checks if the strategy reaches the profit target and provides
    detailed analysis of when the target was reached and the profit stability
    after reaching the target.
    """

    def __init__(self, config: FTMOAccountConfig):
        super().__init__(config)
        self.name = "FTMOProfitTargetValidation"

        if (
            self.config.profit_target_percent <= 0
            or self.config.profit_target_percent >= 100
        ):
            raise ValueError(
                f"profit_target_percent must be between 0 and 100, got {self.config.profit_target_percent}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Check if the strategy reaches the FTMO profit target.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome.
        """
        # Get the equity curve for analysis
        equity = strategy.get_equity_curve()

        if equity.empty:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.config.profit_target_percent,
                explanation=f"No equity data available to evaluate profit target of {self.config.profit_target_percent}%.",
            )

        # Calculate profit percentage at each point
        profit_series = equity - self.config.account_size
        profit_pct_series = (profit_series / self.config.account_size) * 100

        # Find if and when the target was reached
        target_reached = profit_pct_series.max() >= self.config.profit_target_percent

        if target_reached:
            # Find the first time target was reached
            target_reached_idx = profit_pct_series[
                profit_pct_series >= self.config.profit_target_percent
            ].index[0]
            target_reached_time = target_reached_idx.to_pydatetime()
            target_profit_pct = profit_pct_series.loc[target_reached_idx]
            target_profit_amount = profit_series.loc[target_reached_idx]

            # Calculate post-target metrics
            post_target_equity = equity.loc[equity.index >= target_reached_idx]
            if len(post_target_equity) > 1:
                lowest_post_target_pct = (
                    (post_target_equity.min() - self.config.account_size)
                    / self.config.account_size
                ) * 100
                final_profit_pct = profit_pct_series.iloc[-1]
                maintained_target = (
                    lowest_post_target_pct >= self.config.profit_target_percent
                    and final_profit_pct >= self.config.profit_target_percent
                )
            else:
                maintained_target = (
                    True  # Target was reached at the very end, so it's maintained
                )
                lowest_post_target_pct = target_profit_pct
                final_profit_pct = target_profit_pct
        else:
            # Target wasn't reached
            target_reached_time = None
            target_profit_pct = profit_pct_series.max()
            target_profit_amount = profit_series.max()
            maintained_target = False
            lowest_post_target_pct = None
            final_profit_pct = profit_pct_series.iloc[-1]

        # Determine if validation passed - target must be reached
        passed = target_reached

        # Create detailed explanation
        explanation = self._create_explanation(
            passed,
            target_reached_time,
            target_profit_pct,
            target_profit_amount,
            maintained_target,
            lowest_post_target_pct,
            final_profit_pct,
        )

        # The score is the maximum profit percentage achieved
        max_profit_pct = profit_pct_series.max()

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=max_profit_pct,
            threshold=self.config.profit_target_percent,
            explanation=explanation,
        )

    def _create_explanation(
        self,
        passed: bool,
        target_reached_time: Optional[datetime],
        target_profit_pct: float,
        target_profit_amount: float,
        maintained_target: bool,
        lowest_post_target_pct: Optional[float],
        final_profit_pct: float,
    ) -> str:
        """
        Create a detailed explanation message based on profit target analysis.

        Args:
            passed: Whether the validation passed.
            target_reached_time: When the profit target was first reached, if ever.
            target_profit_pct: Profit percentage when target was reached or maximum achieved.
            target_profit_amount: Profit amount when target was reached or maximum achieved.
            maintained_target: Whether the profit remained above target after reaching it.
            lowest_post_target_pct: Lowest profit percentage after reaching target.
            final_profit_pct: Final profit percentage at the end of the testing period.

        Returns:
            Detailed explanation string.
        """
        explanation = (
            f"FTMO profit target requirement: {self.config.profit_target_percent:.2f}% "
            f"({self.config.profit_target_percent/100*self.config.account_size:.2f} {self.config.currency}). "
        )

        if passed:
            # Format target reached time
            time_str = (
                target_reached_time.strftime("%Y-%m-%d %H:%M:%S")
                if target_reached_time
                else "N/A"
            )

            explanation += (
                f"Strategy reached the profit target on {time_str} with a profit of "
                f"{target_profit_pct:.2f}% ({target_profit_amount:.2f} {self.config.currency})."
            )

            # Add information about profit stability after reaching target
            if maintained_target:
                explanation += (
                    f"\nThe profit remained above the target for the remainder of the testing period. "
                    f"Final profit: {final_profit_pct:.2f}% "
                    f"({final_profit_pct/100*self.config.account_size:.2f} {self.config.currency})."
                )
            else:
                explanation += (
                    f"\nWARNING: After reaching the target, the profit dipped to a low of "
                    f"{lowest_post_target_pct:.2f}%. Final profit: {final_profit_pct:.2f}% "
                    f"({final_profit_pct/100*self.config.account_size:.2f} {self.config.currency})."
                )
        else:
            explanation += (
                f"Validation FAILED - Maximum profit achieved was only {target_profit_pct:.2f}% "
                f"({target_profit_amount:.2f} {self.config.currency}), which is "
                f"{self.config.profit_target_percent - target_profit_pct:.2f}% short of the target."
                f"\nFinal profit: {final_profit_pct:.2f}% "
                f"({final_profit_pct/100*self.config.account_size:.2f} {self.config.currency})."
            )

        return explanation


# Register the validation algorithm
register_validation(
    "FTMOProfitTargetValidation", ProfitTargetValidation, FTMOAccountConfig
)
