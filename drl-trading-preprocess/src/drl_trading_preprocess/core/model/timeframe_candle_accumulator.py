"""
Timeframe candle accumulator data model.

Pure data container for accumulating OHLCV data for a specific timeframe period.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from drl_trading_common.model.timeframe import Timeframe


@dataclass
class TimeframeCandleAccumulator:
    """
    Pure data accumulator for building OHLCV candles for a specific timeframe.

    Maintains state for accumulating base timeframe records into
    higher timeframe candles. Contains only data, no business logic.
    """

    timeframe: Timeframe
    current_period_start: Optional[datetime] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: int = 0
    record_count: int = 0

    def is_empty(self) -> bool:
        """Check if accumulator has no data."""
        return self.record_count == 0

    def reset(self) -> None:
        """Reset accumulator for next period."""
        self.current_period_start = None
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        self.volume = 0
        self.record_count = 0
