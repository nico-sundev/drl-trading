"""
Candle accumulator service for OHLCV aggregation business logic.

Provides services for accumulating market data records into higher timeframe candles
with proper period boundary detection and OHLCV aggregation logic.
"""

import logging
from datetime import datetime, timedelta

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.model.resample.timeframe_candle_accumulator import TimeframeCandleAccumulator


class CandleAccumulatorService:
    """
    Service for managing OHLCV candle accumulation logic.

    Handles the business logic for:
    - Period boundary detection
    - OHLCV aggregation rules
    - Candle building and validation
    """

    def __init__(self) -> None:
        """Initialize the accumulator service."""
        self.logger = logging.getLogger(__name__)

    def add_record_to_accumulator(
        self,
        accumulator: TimeframeCandleAccumulator,
        record: MarketDataModel
    ) -> None:
        """
        Add a base timeframe record to the accumulator.

        Args:
            accumulator: The accumulator to update
            record: The market data record to add
        """
        if accumulator.is_empty():
            # First record initializes all values
            accumulator.open_price = record.open_price
            accumulator.high_price = record.high_price
            accumulator.low_price = record.low_price
            accumulator.current_period_start = self.calculate_period_start(
                record.timestamp, accumulator.timeframe
            )
        else:
            # Update high and low
            if accumulator.high_price is None:
                accumulator.high_price = record.high_price
            else:
                accumulator.high_price = max(accumulator.high_price, record.high_price)
            if accumulator.low_price is None:
                accumulator.low_price = record.low_price
            else:
                accumulator.low_price = min(accumulator.low_price, record.low_price)

        # Close price is always the latest record's close
        accumulator.close_price = record.close_price
        accumulator.volume += record.volume
        accumulator.record_count += 1

    def build_candle_from_accumulator(
        self,
        accumulator: TimeframeCandleAccumulator,
        symbol: str
    ) -> MarketDataModel:
        """
        Build a complete OHLCV candle from accumulated data.

        Args:
            accumulator: The accumulator containing aggregated data
            symbol: The symbol for the candle

        Returns:
            Complete market data model representing the aggregated candle

        Raises:
            ValueError: If accumulator is empty
        """
        if accumulator.is_empty():
            raise ValueError("Cannot build candle from empty accumulator")
        if accumulator.current_period_start is None:
            raise ValueError("Accumulator current_period_start is None, cannot build candle")

        return MarketDataModel(
            symbol=symbol,
            timeframe=accumulator.timeframe,
            timestamp=accumulator.current_period_start,
            open_price=accumulator.open_price,
            high_price=accumulator.high_price,
            low_price=accumulator.low_price,
            close_price=accumulator.close_price,
            volume=accumulator.volume
        )

    def calculate_period_start(self, timestamp: datetime, timeframe: Timeframe) -> datetime:
        """
        Calculate the period start time for the given timestamp and timeframe.

        Args:
            timestamp: The timestamp to normalize
            timeframe: The target timeframe

        Returns:
            Normalized timestamp representing the start of the period
        """
        # Normalize to start of period based on timeframe
        if timeframe == Timeframe.MINUTE_5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == Timeframe.MINUTE_15:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == Timeframe.MINUTE_30:
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == Timeframe.HOUR_1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == Timeframe.HOUR_4:
            hour = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == Timeframe.DAY_1:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == Timeframe.WEEK_1:
            # Start of week (Monday)
            days_since_monday = timestamp.weekday()
            week_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            return week_start - timedelta(days=days_since_monday)
        elif timeframe == Timeframe.MONTH_1:
            # Start of month
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default: return timestamp as-is
            self.logger.warning(f"Unknown timeframe {timeframe.value}, returning timestamp as-is")
            return timestamp

    def should_start_new_period(
        self,
        accumulator: TimeframeCandleAccumulator,
        new_record: MarketDataModel
    ) -> bool:
        """
        Determine if a new period should be started based on the new record timestamp.

        Args:
            accumulator: Current accumulator state
            new_record: New record to evaluate

        Returns:
            True if a new period should be started
        """
        if accumulator.is_empty():
            return False

        new_period_start = self.calculate_period_start(
            new_record.timestamp, accumulator.timeframe
        )

        return new_period_start != accumulator.current_period_start
