# --- Profit Factor Validation Config ---
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProfitFactorConfig:
    """
    Configuration for the Profit Factor validation algorithm.

    Profit Factor is the ratio of gross profits to gross losses.
    A ratio greater than 1 indicates a profitable trading system.

    Attributes:
        min_profit_factor: Minimum acceptable profit factor. Default is 1.5.
        include_open_trades: Whether to include open trades in the calculation. Default is False.
        custom_benchmark: Optional custom benchmark value to compare against.
    """

    min_profit_factor: float = 1.5
    include_open_trades: bool = False
    custom_benchmark: Optional[float] = None
