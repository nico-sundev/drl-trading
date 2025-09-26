"""
Tests for CandleAccumulatorService.

This module contains unit tests for the candle accumulation business logic,
including OHLCV aggregation, period boundary detection, and accumulator management.
"""

import pytest
from datetime import datetime

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.model.resample.timeframe_candle_accumulator import TimeframeCandleAccumulator
from drl_trading_preprocess.core.service.candle_accumulator_service import CandleAccumulatorService


class TestCandleAccumulatorService:
    """Test suite for CandleAccumulatorService."""

    @pytest.fixture
    def accumulator_service(self):
        """Create candle accumulator service."""
        return CandleAccumulatorService()

    def test_accumulator_basic_functionality(self, accumulator_service):
        """Test basic OHLCV accumulation using the service."""
        # Given
        accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_5)
        symbol = "EURUSD"

        records = [
            MarketDataModel(
                symbol=symbol,
                timeframe=Timeframe.MINUTE_1,
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.1,
                volume=1000,
            ),
            MarketDataModel(
                symbol=symbol,
                timeframe=Timeframe.MINUTE_1,
                timestamp=datetime(2024, 1, 1, 10, 1, 0),
                open_price=100.1,
                high_price=100.3,
                low_price=99.9,
                close_price=100.2,
                volume=1500,
            ),
        ]

        # When
        for record in records:
            accumulator_service.add_record_to_accumulator(accumulator, record)

        candle = accumulator_service.build_candle_from_accumulator(accumulator, symbol)

        # Then
        assert candle.symbol == symbol
        assert candle.timeframe == Timeframe.MINUTE_5
        assert candle.timestamp == datetime(2024, 1, 1, 10, 0, 0)  # Period start
        assert candle.open_price == 100.0  # First record's open
        assert candle.high_price == 100.3  # Highest high
        assert candle.low_price == 99.8  # Lowest low
        assert candle.close_price == 100.2  # Last record's close
        assert candle.volume == 2500  # Sum of volumes

    def test_accumulator_period_boundary_detection(self, accumulator_service):
        """Test period boundary detection for different timeframes."""
        # Given
        # Test 5-minute boundaries
        time_5m_period1 = datetime(2024, 1, 1, 10, 2, 30)  # Should be 10:00 period
        time_5m_period2 = datetime(2024, 1, 1, 10, 7, 15)  # Should be 10:05 period

        # When
        period1_start = accumulator_service.calculate_period_start(
            time_5m_period1, Timeframe.MINUTE_5
        )
        period2_start = accumulator_service.calculate_period_start(
            time_5m_period2, Timeframe.MINUTE_5
        )

        # Then
        assert period1_start == datetime(2024, 1, 1, 10, 0, 0)
        assert period2_start == datetime(2024, 1, 1, 10, 5, 0)

        # Test 1-hour boundaries
        time_1h_period1 = datetime(2024, 1, 1, 10, 45, 30)  # Should be 10:00 period
        time_1h_period2 = datetime(2024, 1, 1, 11, 15, 45)  # Should be 11:00 period

        period1_start_1h = accumulator_service.calculate_period_start(
            time_1h_period1, Timeframe.HOUR_1
        )
        period2_start_1h = accumulator_service.calculate_period_start(
            time_1h_period2, Timeframe.HOUR_1
        )

        assert period1_start_1h == datetime(2024, 1, 1, 10, 0, 0)
        assert period2_start_1h == datetime(2024, 1, 1, 11, 0, 0)

    def test_accumulator_reset_functionality(self, accumulator_service):
        """Test accumulator reset between periods."""
        # Given
        accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_5)

        record = MarketDataModel(
            symbol="EURUSD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            open_price=100.0,
            high_price=100.2,
            low_price=99.8,
            close_price=100.1,
            volume=1000,
        )

        accumulator_service.add_record_to_accumulator(accumulator, record)
        assert not accumulator.is_empty()
        assert accumulator.volume == 1000

        # When
        accumulator.reset()

        # Then
        assert accumulator.is_empty()
        assert accumulator.volume == 0
        assert accumulator.open_price is None
        assert accumulator.current_period_start is None

    def test_should_start_new_period_detection(self, accumulator_service):
        """Test new period detection logic."""
        # Given
        accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_5)

        # First record - should not trigger new period (accumulator is empty)
        first_record = MarketDataModel(
            symbol="EURUSD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            open_price=100.0,
            high_price=100.2,
            low_price=99.8,
            close_price=100.1,
            volume=1000,
        )

        # When
        should_start_new = accumulator_service.should_start_new_period(accumulator, first_record)

        # Then
        assert not should_start_new  # Empty accumulator should not start new period

        # Add first record
        accumulator_service.add_record_to_accumulator(accumulator, first_record)

        # Second record in same period - should not trigger new period
        same_period_record = MarketDataModel(
            symbol="EURUSD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 10, 4, 0),  # Still in 10:00-10:04 period
            open_price=100.1,
            high_price=100.3,
            low_price=99.9,
            close_price=100.2,
            volume=1500,
        )

        should_start_new = accumulator_service.should_start_new_period(accumulator, same_period_record)
        assert not should_start_new

        # Third record in new period - should trigger new period
        new_period_record = MarketDataModel(
            symbol="EURUSD",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime(2024, 1, 1, 10, 5, 0),  # New period: 10:05-10:09
            open_price=100.2,
            high_price=100.4,
            low_price=100.0,
            close_price=100.3,
            volume=2000,
        )

        should_start_new = accumulator_service.should_start_new_period(accumulator, new_period_record)
        assert should_start_new

    def test_build_candle_validation(self, accumulator_service):
        """Test candle building validation logic."""
        # Given
        empty_accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_5)

        # When & Then
        with pytest.raises(ValueError, match="Cannot build candle from empty accumulator"):
            accumulator_service.build_candle_from_accumulator(empty_accumulator, "EURUSD")

    def test_calculate_period_start_edge_cases(self, accumulator_service):
        """Test period start calculation for various timeframes."""
        # Given
        base_timestamp = datetime(2024, 1, 1, 14, 33, 45)  # 2:33:45 PM

        # When & Then - Test various timeframes

        # 15-minute periods: should round down to 14:30
        period_15m = accumulator_service.calculate_period_start(base_timestamp, Timeframe.MINUTE_15)
        assert period_15m == datetime(2024, 1, 1, 14, 30, 0)

        # 30-minute periods: should round down to 14:30
        period_30m = accumulator_service.calculate_period_start(base_timestamp, Timeframe.MINUTE_30)
        assert period_30m == datetime(2024, 1, 1, 14, 30, 0)

        # 4-hour periods: should round down to 12:00 (14 // 4 = 3, 3 * 4 = 12)
        period_4h = accumulator_service.calculate_period_start(base_timestamp, Timeframe.HOUR_4)
        assert period_4h == datetime(2024, 1, 1, 12, 0, 0)

        # Daily periods: should round down to start of day
        period_1d = accumulator_service.calculate_period_start(base_timestamp, Timeframe.DAY_1)
        assert period_1d == datetime(2024, 1, 1, 0, 0, 0)

        # Weekly periods: should round down to Monday (January 1, 2024 was a Monday)
        period_1w = accumulator_service.calculate_period_start(base_timestamp, Timeframe.WEEK_1)
        assert period_1w == datetime(2024, 1, 1, 0, 0, 0)

        # Monthly periods: should round down to start of month
        period_1m = accumulator_service.calculate_period_start(base_timestamp, Timeframe.MONTH_1)
        assert period_1m == datetime(2024, 1, 1, 0, 0, 0)
