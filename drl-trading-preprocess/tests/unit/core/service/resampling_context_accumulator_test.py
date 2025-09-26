"""Unit tests for ResamplingContext accumulator methods and state management."""

from datetime import datetime

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.resample.resampling_context import (
    AccumulatorState,
    ResamplingContext
)
from drl_trading_preprocess.core.model.resample.timeframe_candle_accumulator import TimeframeCandleAccumulator


class TestAccumulatorStateConversion:
    """Test AccumulatorState conversion methods."""

    def test_to_accumulator_conversion(self) -> None:
        """Test conversion from AccumulatorState to TimeframeCandleAccumulator."""
        # Given
        accumulator_state = AccumulatorState(
            symbol="BTCUSDT",
            target_timeframe=Timeframe.MINUTE_5,
            current_period_start=datetime(2024, 1, 1, 10, 0, 0),
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.5,
            volume=1500,
            record_count=3
        )

        # When
        accumulator = accumulator_state.to_accumulator()

        # Then
        assert isinstance(accumulator, TimeframeCandleAccumulator)
        assert accumulator.timeframe == Timeframe.MINUTE_5
        assert accumulator.current_period_start == datetime(2024, 1, 1, 10, 0, 0)
        assert accumulator.open_price == 100.0
        assert accumulator.high_price == 102.0
        assert accumulator.low_price == 99.0
        assert accumulator.close_price == 101.5
        assert accumulator.volume == 1500
        assert accumulator.record_count == 3

    def test_from_accumulator_conversion(self) -> None:
        """Test creation of AccumulatorState from TimeframeCandleAccumulator."""
        # Given
        accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_1)
        accumulator.current_period_start = datetime(2024, 1, 1, 9, 0, 0)
        accumulator.open_price = 50.0
        accumulator.high_price = 52.0
        accumulator.low_price = 49.0
        accumulator.close_price = 51.0
        accumulator.volume = 2000
        accumulator.record_count = 5

        # When
        accumulator_state = AccumulatorState.from_accumulator(accumulator, "ETHUSDT")

        # Then
        assert accumulator_state.symbol == "ETHUSDT"
        assert accumulator_state.target_timeframe == Timeframe.MINUTE_1
        assert accumulator_state.current_period_start == datetime(2024, 1, 1, 9, 0, 0)
        assert accumulator_state.open_price == 50.0
        assert accumulator_state.high_price == 52.0
        assert accumulator_state.low_price == 49.0
        assert accumulator_state.close_price == 51.0
        assert accumulator_state.volume == 2000
        assert accumulator_state.record_count == 5

    def test_round_trip_conversion_preserves_data(self) -> None:
        """Test that converting to accumulator and back preserves all data."""
        # Given
        original_state = AccumulatorState(
            symbol="ADAUSDT",
            target_timeframe=Timeframe.MINUTE_15,
            current_period_start=datetime(2024, 1, 1, 12, 0, 0),
            open_price=0.5,
            high_price=0.52,
            low_price=0.48,
            close_price=0.51,
            volume=10000,
            record_count=10
        )

        # When
        accumulator = original_state.to_accumulator()
        restored_state = AccumulatorState.from_accumulator(accumulator, "ADAUSDT")

        # Then
        assert restored_state.symbol == original_state.symbol
        assert restored_state.target_timeframe == original_state.target_timeframe
        assert restored_state.current_period_start == original_state.current_period_start
        assert restored_state.open_price == original_state.open_price
        assert restored_state.high_price == original_state.high_price
        assert restored_state.low_price == original_state.low_price
        assert restored_state.close_price == original_state.close_price
        assert restored_state.volume == original_state.volume
        assert restored_state.record_count == original_state.record_count

    def test_to_accumulator_with_none_values(self) -> None:
        """Test accumulator conversion with None price values."""
        # Given
        accumulator_state = AccumulatorState(
            symbol="BTCUSDT",
            target_timeframe=Timeframe.MINUTE_5,
            current_period_start=datetime(2024, 1, 1, 10, 0, 0),
            # All price values default to None
            volume=0,
            record_count=0
        )

        # When
        accumulator = accumulator_state.to_accumulator()

        # Then
        assert accumulator.timeframe == Timeframe.MINUTE_5
        assert accumulator.current_period_start == datetime(2024, 1, 1, 10, 0, 0)
        assert accumulator.open_price is None
        assert accumulator.high_price is None
        assert accumulator.low_price is None
        assert accumulator.close_price is None
        assert accumulator.volume == 0
        assert accumulator.record_count == 0


class TestResamplingContextAccumulatorManagement:
    """Test ResamplingContext accumulator persistence methods."""

    def test_get_accumulator_creates_new_if_not_exists(self) -> None:
        """Test that get_accumulator creates new accumulator if it doesn't exist."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # When
        accumulator = context.get_accumulator("BTCUSDT", Timeframe.MINUTE_5)

        # Then
        assert isinstance(accumulator, TimeframeCandleAccumulator)
        assert accumulator.timeframe == Timeframe.MINUTE_5

    def test_get_accumulator_restores_existing_state(self) -> None:
        """Test that get_accumulator restores accumulator from persisted state."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # Create and persist accumulator state
        original_accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_1)
        original_accumulator.current_period_start = datetime(2024, 1, 1, 10, 0, 0)
        original_accumulator.open_price = 100.0
        original_accumulator.high_price = 101.0
        original_accumulator.low_price = 99.0
        original_accumulator.volume = 1000

        context.persist_accumulator_state("BTCUSDT", original_accumulator)

        # When
        restored_accumulator = context.get_accumulator("BTCUSDT", Timeframe.MINUTE_1)

        # Then
        assert restored_accumulator.timeframe == Timeframe.MINUTE_1
        assert restored_accumulator.current_period_start == datetime(2024, 1, 1, 10, 0, 0)
        assert restored_accumulator.open_price == 100.0
        assert restored_accumulator.high_price == 101.0
        assert restored_accumulator.low_price == 99.0
        assert restored_accumulator.volume == 1000

    def test_persist_accumulator_state_stores_state(self) -> None:
        """Test that persist_accumulator_state correctly stores accumulator state."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)
        accumulator = TimeframeCandleAccumulator(timeframe=Timeframe.MINUTE_5)
        accumulator.current_period_start = datetime(2024, 1, 1, 15, 0, 0)
        accumulator.open_price = 200.0
        accumulator.close_price = 205.0
        accumulator.volume = 5000
        accumulator.record_count = 8

        # When
        context.persist_accumulator_state("ETHUSDT", accumulator)

        # Then
        # Verify state is stored internally
        assert "ETHUSDT" in context._accumulator_states
        assert Timeframe.MINUTE_5 in context._accumulator_states["ETHUSDT"]

        stored_state = context._accumulator_states["ETHUSDT"][Timeframe.MINUTE_5]
        assert stored_state.symbol == "ETHUSDT"
        assert stored_state.target_timeframe == Timeframe.MINUTE_5
        assert stored_state.current_period_start == datetime(2024, 1, 1, 15, 0, 0)
        assert stored_state.open_price == 200.0
        assert stored_state.close_price == 205.0
        assert stored_state.volume == 5000
        assert stored_state.record_count == 8

    def test_get_accumulator_multiple_symbols_and_timeframes(self) -> None:
        """Test accumulator management across multiple symbols and timeframes."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # When - Create accumulators for different symbols and timeframes
        btc_1m = context.get_accumulator("BTCUSDT", Timeframe.MINUTE_1)
        btc_5m = context.get_accumulator("BTCUSDT", Timeframe.MINUTE_5)
        eth_1m = context.get_accumulator("ETHUSDT", Timeframe.MINUTE_1)

        # Then
        assert btc_1m.timeframe == Timeframe.MINUTE_1
        assert btc_5m.timeframe == Timeframe.MINUTE_5
        assert eth_1m.timeframe == Timeframe.MINUTE_1

        # Verify they are independent
        assert btc_1m is not btc_5m
        assert btc_1m is not eth_1m
        assert btc_5m is not eth_1m


class TestResamplingContextStateManagement:
    """Test additional ResamplingContext state management methods."""

    def test_get_symbols_for_processing(self) -> None:
        """Test getting list of symbols currently being processed."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # Add some symbols
        context.update_last_processed_timestamp("BTCUSDT", Timeframe.MINUTE_1, datetime.now())
        context.update_last_processed_timestamp("ETHUSDT", Timeframe.MINUTE_5, datetime.now())
        context.update_last_processed_timestamp("ADAUSDT", Timeframe.MINUTE_1, datetime.now())

        # When
        symbols = context.get_symbols_for_processing()

        # Then
        assert set(symbols) == {"BTCUSDT", "ETHUSDT", "ADAUSDT"}

    def test_get_processing_stats(self) -> None:
        """Test getting processing statistics."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # Add some statistics
        context.increment_stats("BTCUSDT", Timeframe.MINUTE_1, 100, 20)
        context.increment_stats("BTCUSDT", Timeframe.MINUTE_5, 50, 10)
        context.increment_stats("ETHUSDT", Timeframe.MINUTE_1, 75, 15)

        # When
        stats = context.get_processing_stats()

        # Then
        assert "BTCUSDT" in stats
        assert "ETHUSDT" in stats

        btc_stats = stats["BTCUSDT"]
        assert "1m" in btc_stats
        assert "5m" in btc_stats

        # Verify specific stats
        btc_1m_stats = btc_stats["1m"]
        assert btc_1m_stats["records_processed"] == 100
        assert btc_1m_stats["candles_generated"] == 20

    def test_clean_inactive_symbols_removes_specified_symbols(self) -> None:
        """Test that clean_inactive_symbols removes symbols NOT in active list."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # Add multiple symbols
        symbols_to_add = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        for symbol in symbols_to_add:
            context.update_last_processed_timestamp(symbol, Timeframe.MINUTE_1, datetime.now())
            context.increment_stats(symbol, Timeframe.MINUTE_1, 10, 2)

        # When - Keep only these symbols active
        active_symbols = ["BTCUSDT", "ADAUSDT"]  # ETHUSDT and DOTUSDT should be removed
        context.clean_inactive_symbols(active_symbols)

        # Then
        remaining_symbols = context.get_symbols_for_processing()
        assert "BTCUSDT" in remaining_symbols
        assert "ADAUSDT" in remaining_symbols
        assert "ETHUSDT" not in remaining_symbols
        assert "DOTUSDT" not in remaining_symbols

    def test_clean_inactive_symbols_removes_accumulator_states(self) -> None:
        """Test that clean_inactive_symbols also removes accumulator states."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=100)

        # Create accumulators and symbol states for symbols
        context.get_accumulator("BTCUSDT", Timeframe.MINUTE_1)
        context.get_accumulator("ETHUSDT", Timeframe.MINUTE_1)

        # Create symbol states by calling update_last_processed_timestamp - this is what makes symbols eligible for cleanup
        from datetime import datetime
        context.update_last_processed_timestamp("BTCUSDT", Timeframe.MINUTE_1, datetime.now())
        context.update_last_processed_timestamp("ETHUSDT", Timeframe.MINUTE_1, datetime.now())

        # Verify they exist
        assert "BTCUSDT" in context._active_accumulators
        assert "ETHUSDT" in context._active_accumulators
        assert "BTCUSDT" in context._symbol_states
        assert "ETHUSDT" in context._symbol_states

        # When - Keep only BTCUSDT active
        context.clean_inactive_symbols(["BTCUSDT"])

        # Then
        assert "BTCUSDT" in context._active_accumulators
        assert "ETHUSDT" not in context._active_accumulators
        assert "BTCUSDT" in context._symbol_states
        assert "ETHUSDT" not in context._symbol_states

    def test_memory_management_with_max_symbols_limit(self) -> None:
        """Test that context can track symbols up to limit."""
        # Given
        context = ResamplingContext(max_symbols_in_memory=2)

        # When - Add symbols up to the limit
        context.update_last_processed_timestamp("SYMBOL1", Timeframe.MINUTE_1, datetime.now())
        context.update_last_processed_timestamp("SYMBOL2", Timeframe.MINUTE_1, datetime.now())

        # Then - Should have 2 symbols
        symbols = context.get_symbols_for_processing()
        assert len(symbols) == 2

        # When - Add one more symbol (limit is guidance, not enforcement in current implementation)
        context.update_last_processed_timestamp("SYMBOL3", Timeframe.MINUTE_1, datetime.now())

        # Then - Implementation allows going over limit, relies on external cleanup
        symbols_after = context.get_symbols_for_processing()
        assert len(symbols_after) == 3  # Current implementation doesn't auto-cleanup
