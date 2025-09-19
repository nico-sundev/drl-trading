"""
Market data domain model for business operations.

This module defines the MarketDataModel which represents market data from a business
perspective, free from database and persistence concerns. Used throughout the core
domain layer for business logic and service interfaces.
"""

from datetime import datetime
from typing import Optional

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.model.timeframe import Timeframe


class MarketDataModel(BaseSchema):
    """
    Business domain model for market data.

    Represents a single market data point with OHLCV information.
    This is the canonical business representation used throughout the core domain,
    independent of database storage or external data formats.
    """

    # Business identifiers
    symbol: str
    timeframe: Timeframe
    timestamp: datetime

    # OHLCV market data
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int = 0

    # Business metadata
    created_at: Optional[datetime] = None

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        return (
            f"MarketDataModel(symbol={self.symbol}, timeframe={self.timeframe}, "
            f"timestamp={self.timestamp}, close_price={self.close_price})"
        )

    def __repr__(self) -> str:
        """Detailed representation for development."""
        return (
            f"MarketDataModel(symbol='{self.symbol}', timeframe={self.timeframe}, "
            f"timestamp={self.timestamp}, open={self.open_price}, "
            f"high={self.high_price}, low={self.low_price}, "
            f"close={self.close_price}, volume={self.volume})"
        )

    def validate_ohlc_logic(self) -> bool:
        """
        Validate OHLC price logic.

        Returns:
            bool: True if OHLC prices are logically consistent
        """
        return (
            self.high_price >= max(self.open_price, self.close_price) and
            self.low_price <= min(self.open_price, self.close_price) and
            self.high_price >= self.low_price and
            self.volume >= 0
        )
