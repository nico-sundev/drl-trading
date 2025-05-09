from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """
    Represents a single trade with its details and performance metrics.

    This class encapsulates all relevant information about a trade that can be
    used for analysis, validation, and reporting purposes.

    Attributes:
        symbol: The trading instrument symbol (e.g., "EURUSD")
        entry_time: Timestamp when the trade was opened
        exit_time: Timestamp when the trade was closed (None if still open)
        entry_price: Price at which the position was opened
        exit_price: Price at which the position was closed (None if still open)
        direction: Trade direction: 1 for long/buy, -1 for short/sell
        size: Position size (e.g., lot size or number of contracts)
        profit: Realized profit/loss of the trade (negative for losses)
        pips: Number of pips gained or lost
        cumulative_equity: Account equity after this trade
        stop_loss: Stop loss price level (if set)
        take_profit: Take profit price level (if set)
        commission: Commission paid for this trade
        swap: Swap/rollover fees for this trade
        tags: Optional categorization tags for the trade
    """

    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    direction: int = 1  # 1 for long, -1 for short
    size: float = 0.0
    profit: float = 0.0
    pips: float = 0.0
    cumulative_equity: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    tags: Optional[dict] = None

    @property
    def is_open(self) -> bool:
        """Return True if the trade is still open (no exit time)."""
        return self.exit_time is None

    @property
    def is_winning(self) -> bool:
        """Return True if this is a winning trade (profit > 0)."""
        return self.profit > 0

    @property
    def duration(self) -> Optional[float]:
        """Return trade duration in seconds, or None if the trade is still open."""
        if self.is_open:
            return None
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """
        Calculate risk-reward ratio if stop loss was set.

        Returns:
            Float representing the risk-reward ratio, or None if stop loss wasn't set.
        """
        if self.stop_loss is None or self.is_open:
            return None

        potential_loss = abs(self.entry_price - self.stop_loss) * self.direction
        if potential_loss == 0:
            return None

        if self.exit_price:
            realized_reward = (self.exit_price - self.entry_price) * self.direction
            return realized_reward / potential_loss
        return None
