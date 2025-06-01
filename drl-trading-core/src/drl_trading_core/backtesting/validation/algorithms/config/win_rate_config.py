# --- Win Rate Validation Config ---
from dataclasses import dataclass


@dataclass(frozen=True)
class WinRateConfig:
    """
    Configuration for the Win Rate validation algorithm.

    Win Rate is the percentage of trades that result in a profit.
    This validator checks if a strategy maintains a minimum win rate.

    Attributes:
        min_win_rate: Minimum acceptable win rate (0-100). Default is 40.0.
        min_trade_count: Minimum number of trades required for a valid win rate calculation. Default is 10.
        include_open_trades: Whether to include open trades in the calculation. Default is False.
        require_profit: Whether to additionally require the strategy to be profitable overall. Default is True.
    """

    min_win_rate: float = 40.0  # percentage
    min_trade_count: int = 10
    include_open_trades: bool = False
    require_profit: bool = True
