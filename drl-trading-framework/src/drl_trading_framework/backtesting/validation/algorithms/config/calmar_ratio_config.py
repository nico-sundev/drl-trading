# --- Calmar Ratio Validation Config ---
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CalmarRatioConfig:
    """
    Configuration for the Calmar Ratio validation algorithm.

    The Calmar Ratio is a risk-adjusted performance measure that divides the average annual
    rate of return by the maximum drawdown for a program. Higher values indicate better
    risk-adjusted performance.

    Attributes:
        min_calmar: Minimum acceptable Calmar Ratio. Default is 0.5.
        lookback_years: Number of years to consider for the calculation. Default is 3 years.
        annualization_factor: Factor to annualize returns based on data frequency.
                             Default is 252 (trading days in a year).
        custom_benchmark: Optional custom benchmark value to compare against.
    """

    min_calmar: float = 0.5
    lookback_years: float = 3.0
    annualization_factor: int = 252
    custom_benchmark: Optional[float] = None
