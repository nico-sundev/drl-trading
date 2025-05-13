# --- Max Consecutive Losses Validation Config ---
from dataclasses import dataclass


@dataclass(frozen=True)
class MaxConsecutiveLossesConfig:
    """
    Configuration for the Maximum Consecutive Losses validation algorithm.

    This validator checks if a strategy exceeds a maximum allowed number of consecutive
    losing trades, which is an important risk management measure.

    Attributes:
        max_consecutive_losses: Maximum number of consecutive losing trades allowed. Default is 5.
        consider_drawdown: Whether to also consider drawdown amount during consecutive losses. Default is False.
        max_drawdown_during_streak: Maximum acceptable drawdown percentage during a losing streak. Default is 15.0.
        include_open_trades: Whether to include open trades in the calculation. Default is False.
    """

    max_consecutive_losses: int = 5
    consider_drawdown: bool = False
    max_drawdown_during_streak: float = 15.0  # percentage
    include_open_trades: bool = False
