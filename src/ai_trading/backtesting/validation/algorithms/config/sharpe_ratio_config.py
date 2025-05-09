# --- Example: Sharpe Ratio Validation ---
from dataclasses import dataclass


@dataclass(frozen=True)
class SharpeRatioConfig:
    min_sharpe: float = 1.0
    window: int = 252
