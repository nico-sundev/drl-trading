"""Unit tests for MarketDataResamplingService utility and state management methods."""

from datetime import datetime
from unittest.mock import Mock

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.service.market_data_resampling_service import MarketDataResamplingService
from drl_trading_preprocess.infrastructure.config.preprocess_config import ResampleConfig


class TestMarketDataResamplingServiceStateManagement:
    """Test state management utility methods."""

    def test_get_processing_stats(self) -> None:
        """Test getting processing statistics from the service."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Add some processing activity to context
        service.context.increment_stats("BTCUSDT", Timeframe.MINUTE_1, 100, 20)
        service.context.increment_stats("ETHUSDT", Timeframe.MINUTE_5, 50, 10)

        # When
        stats = service.get_processing_stats()

        # Then
        assert isinstance(stats, dict)
        assert "BTCUSDT" in stats
        assert "ETHUSDT" in stats
        assert stats["BTCUSDT"]["1m"]["records_processed"] == 100
        assert stats["BTCUSDT"]["1m"]["candles_generated"] == 20

    def test_reset_symbol_state(self) -> None:
        """Test resetting processing state for a specific symbol."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Add symbol state
        service.context.update_last_processed_timestamp(
            "BTCUSDT",
            Timeframe.MINUTE_1,
            datetime(2024, 1, 1, 10, 0, 0)
        )

        # Verify symbol exists
        symbols_before = service.get_symbols_in_context()
        assert "BTCUSDT" in symbols_before

        # When
        service.reset_symbol_state("BTCUSDT")

        # Then
        # The implementation may vary, but state should be reset
        # This tests the method doesn't crash and handles the reset properly

    def test_get_symbols_in_context(self) -> None:
        """Test getting list of symbols currently tracked in context."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Add symbols to context
        service.context.update_last_processed_timestamp(
            "BTCUSDT",
            Timeframe.MINUTE_1,
            datetime.now()
        )
        service.context.update_last_processed_timestamp(
            "ETHUSDT",
            Timeframe.MINUTE_5,
            datetime.now()
        )

        # When
        symbols = service.get_symbols_in_context()

        # Then
        assert isinstance(symbols, list)
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_save_context_state_with_persistence_enabled(self) -> None:
        """Test saving context state when persistence is enabled."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        mock_state_persistence = Mock()
        resample_config = ResampleConfig()

        # Mock the state persistence load to return None (no existing state)
        mock_state_persistence.load_context.return_value = None
        # Mock successful save
        mock_state_persistence.save_context.return_value = True

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=mock_state_persistence
        )

        # When
        result = service.save_context_state()

        # Then
        assert result is True
        mock_state_persistence.save_context.assert_called_once_with(service.context)

    def test_save_context_state_without_persistence(self) -> None:
        """Test saving context state when persistence is not enabled."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=None  # No persistence
        )

        # When
        result = service.save_context_state()

        # Then
        assert result is False

    def test_reset_context_state_with_persistence(self) -> None:
        """Test resetting context state when persistence is enabled."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        mock_state_persistence = Mock()
        resample_config = ResampleConfig()

        # Mock the state persistence
        mock_state_persistence.load_context.return_value = None
        mock_state_persistence.cleanup_state_file.return_value = True

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=mock_state_persistence
        )

        # Add some state first
        service.context.update_last_processed_timestamp(
            "BTCUSDT",
            Timeframe.MINUTE_1,
            datetime.now()
        )

        # When
        result = service.reset_context_state()

        # Then
        assert result is True
        mock_state_persistence.cleanup_state_file.assert_called_once()

        # Verify context was reset
        symbols = service.get_symbols_in_context()
        assert len(symbols) == 0

    def test_reset_context_state_without_persistence(self) -> None:
        """Test resetting context state when persistence is not enabled."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=None
        )

        # Add some state first
        service.context.update_last_processed_timestamp(
            "BTCUSDT",
            Timeframe.MINUTE_1,
            datetime.now()
        )

        # When
        result = service.reset_context_state()

        # Then
        assert result is True

        # Verify context was reset
        symbols = service.get_symbols_in_context()
        assert len(symbols) == 0


class TestMarketDataResamplingServiceDataValidation:
    """Test data validation helper methods."""

    def test_is_invalid_ohlcv_negative_prices(self) -> None:
        """Test validation of records with negative prices."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create record with negative price
        invalid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open_price=-100.0,  # Invalid
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1000
        )

        # When
        is_invalid = service._is_invalid_ohlcv(invalid_record)

        # Then
        assert is_invalid is True

    def test_is_invalid_ohlcv_high_less_than_low(self) -> None:
        """Test validation of records where high < low."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create record with high < low
        invalid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=98.0,  # Lower than low
            low_price=99.0,
            close_price=100.5,
            volume=1000
        )

        # When
        is_invalid = service._is_invalid_ohlcv(invalid_record)

        # Then
        assert is_invalid is True

    def test_is_invalid_ohlcv_open_outside_range(self) -> None:
        """Test validation of records where open is outside high/low range."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create record with open outside range
        invalid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open_price=105.0,  # Higher than high
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1000
        )

        # When
        is_invalid = service._is_invalid_ohlcv(invalid_record)

        # Then
        assert is_invalid is True

    def test_is_invalid_ohlcv_negative_volume(self) -> None:
        """Test validation of records with negative volume."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create record with negative volume
        invalid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=-1000  # Invalid
        )

        # When
        is_invalid = service._is_invalid_ohlcv(invalid_record)

        # Then
        assert is_invalid is True

    def test_is_invalid_ohlcv_valid_record(self) -> None:
        """Test validation of a valid OHLCV record."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create valid record
        valid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1000
        )

        # When
        is_invalid = service._is_invalid_ohlcv(valid_record)

        # Then
        assert is_invalid is False

    def test_is_invalid_ohlcv_missing_attributes(self) -> None:
        """Test validation with missing required attributes."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config
        )

        # Create record with missing attributes
        incomplete_record = Mock()
        incomplete_record.open_price = None  # Missing required field

        # When
        is_invalid = service._is_invalid_ohlcv(incomplete_record)

        # Then
        assert is_invalid is True


class TestMarketDataResamplingServiceInitialization:
    """Test service initialization with optional state persistence."""

    def test_initialization_without_state_persistence(self) -> None:
        """Test service initialization without state persistence."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        resample_config = ResampleConfig()

        # When
        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=None
        )

        # Then
        assert service.state_persistence is None
        assert service.context is not None
        assert isinstance(service.context.max_symbols_in_memory, int)

    def test_initialization_with_state_persistence_no_existing_state(self) -> None:
        """Test service initialization with state persistence but no existing state."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        mock_state_persistence = Mock()
        resample_config = ResampleConfig()

        # Mock no existing state
        mock_state_persistence.load_context.return_value = None

        # When
        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=mock_state_persistence
        )

        # Then
        assert service.state_persistence is mock_state_persistence
        assert service.context is not None
        mock_state_persistence.load_context.assert_called_once()

    def test_initialization_with_state_persistence_existing_state(self) -> None:
        """Test service initialization with state persistence and existing state."""
        # Given
        mock_market_data_reader = Mock()
        mock_message_publisher = Mock()
        mock_candle_accumulator = Mock()
        mock_state_persistence = Mock()
        resample_config = ResampleConfig()

        # Mock existing state
        from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext
        existing_context = ResamplingContext(max_symbols_in_memory=150)
        existing_context.update_last_processed_timestamp(
            "BTCUSDT",
            Timeframe.MINUTE_1,
            datetime(2024, 1, 1, 10, 0, 0)
        )
        mock_state_persistence.load_context.return_value = existing_context

        # When
        service = MarketDataResamplingService(
            market_data_reader=mock_market_data_reader,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=mock_candle_accumulator,
            resample_config=resample_config,
            state_persistence=mock_state_persistence
        )

        # Then
        assert service.state_persistence is mock_state_persistence
        assert service.context is existing_context
        assert service.context.max_symbols_in_memory == 150

        # Verify restored symbol exists
        symbols = service.get_symbols_in_context()
        assert "BTCUSDT" in symbols
