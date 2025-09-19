"""Unit tests for MarketDataMapper.

Tests the mapping functionality between MarketDataEntity and MarketDataModel,
including error handling and edge cases.
"""

import pytest
from drl_trading_adapter.adapter.database.mapper.market_data_mapper import MarketDataMapper
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_common.model.timeframe import Timeframe
from conftest import (
    TEST_SYMBOL, TEST_TIMEFRAME, TEST_TIMESTAMP, TEST_OPEN, TEST_HIGH,
    TEST_LOW, TEST_CLOSE, TEST_VOLUME, ZERO_VOLUME, NEGATIVE_PRICE, VERY_LARGE_VOLUME
)


class TestMarketDataMapperEntityToModel:
    """Test suite for MarketDataMapper.entity_to_model() method."""

    def test_entity_to_model_success(self, sample_market_data_entity: MarketDataEntity) -> None:
        """Test successful conversion from entity to model."""
        # Given
        entity = sample_market_data_entity

        # When
        result = MarketDataMapper.entity_to_model(entity)

        # Then
        assert isinstance(result, MarketDataModel)
        assert result.symbol == entity.symbol
        assert result.timeframe.value == entity.timeframe
        assert result.timestamp == entity.timestamp
        assert result.open_price == entity.open_price
        assert result.high_price == entity.high_price
        assert result.low_price == entity.low_price
        assert result.close_price == entity.close_price
        assert result.volume == entity.volume

    def test_entity_to_model_with_none_entity(self) -> None:
        """Test entity_to_model with None input raises ValueError."""
        # Given
        entity = None

        # When & Then
        with pytest.raises(ValueError, match="Entity cannot be None"):
            MarketDataMapper.entity_to_model(entity)

    def test_entity_to_model_with_missing_required_fields(self) -> None:
        """Test entity_to_model with missing required fields raises ValueError."""
        # Given
        entity = MarketDataEntity()
        entity.symbol = TEST_SYMBOL
        # Intentionally leaving other required fields as None

        # When & Then
        with pytest.raises(ValueError, match="Entity has None values in required fields"):
            MarketDataMapper.entity_to_model(entity)

    @pytest.mark.parametrize("symbol,timeframe,volume", [
        (TEST_SYMBOL, TEST_TIMEFRAME, ZERO_VOLUME),
        ("ABC", TEST_TIMEFRAME, TEST_VOLUME),  # Different but valid symbol
        (TEST_SYMBOL, "1h", TEST_VOLUME),  # Different but valid timeframe
        (TEST_SYMBOL, TEST_TIMEFRAME, VERY_LARGE_VOLUME)
    ])
    def test_entity_to_model_edge_cases(self, symbol: str, timeframe: str, volume: int) -> None:
        """Test entity_to_model with edge case values."""
        # Given
        entity = MarketDataEntity()
        entity.symbol = symbol
        entity.timeframe = timeframe
        entity.timestamp = TEST_TIMESTAMP
        entity.open_price = TEST_OPEN
        entity.high_price = TEST_HIGH
        entity.low_price = TEST_LOW
        entity.close_price = TEST_CLOSE
        entity.volume = volume

        # When
        result = MarketDataMapper.entity_to_model(entity)

        # Then
        assert result.symbol == symbol
        assert result.timeframe.value == timeframe  # Compare with the enum value
        assert result.volume == volume

    def test_entity_to_model_with_empty_symbol_succeeds(self) -> None:
        """Test entity_to_model with empty symbol succeeds (empty string is valid)."""
        # Given
        entity = MarketDataEntity()
        entity.symbol = ""
        entity.timeframe = TEST_TIMEFRAME
        entity.timestamp = TEST_TIMESTAMP
        entity.open_price = TEST_OPEN
        entity.high_price = TEST_HIGH
        entity.low_price = TEST_LOW
        entity.close_price = TEST_CLOSE
        entity.volume = TEST_VOLUME

        # When
        result = MarketDataMapper.entity_to_model(entity)

        # Then
        assert result.symbol == ""
        assert result.timeframe.value == TEST_TIMEFRAME

    def test_entity_to_model_with_invalid_timeframe_raises_error(self) -> None:
        """Test entity_to_model with invalid timeframe raises appropriate error."""
        # Given
        entity = MarketDataEntity()
        entity.symbol = TEST_SYMBOL
        entity.timeframe = ""  # Invalid timeframe
        entity.timestamp = TEST_TIMESTAMP
        entity.open_price = TEST_OPEN
        entity.high_price = TEST_HIGH
        entity.low_price = TEST_LOW
        entity.close_price = TEST_CLOSE
        entity.volume = TEST_VOLUME

        # When & Then
        with pytest.raises(ValueError, match="Failed to convert entity to model"):
            MarketDataMapper.entity_to_model(entity)


class TestMarketDataMapperModelToEntity:
    """Test suite for MarketDataMapper.model_to_entity() method."""

    def test_model_to_entity_success(self, sample_market_data_model: MarketDataModel) -> None:
        """Test successful conversion from model to entity."""
        # Given
        model = sample_market_data_model

        # When
        result = MarketDataMapper.model_to_entity(model)

        # Then
        assert isinstance(result, MarketDataEntity)
        assert result.symbol == model.symbol
        assert result.timeframe == model.timeframe.value  # Entity stores timeframe as string
        assert result.timestamp == model.timestamp
        assert result.open_price == model.open_price
        assert result.high_price == model.high_price
        assert result.low_price == model.low_price
        assert result.close_price == model.close_price
        assert result.volume == model.volume

    def test_model_to_entity_with_none_model(self) -> None:
        """Test model_to_entity with None input raises ValueError."""
        # Given
        model = None

        # When & Then
        with pytest.raises(ValueError, match="Model cannot be None"):
            MarketDataMapper.model_to_entity(model)

    def test_model_to_entity_with_invalid_model_type(self) -> None:
        """Test model_to_entity with invalid model type raises ValueError."""
        # Given
        invalid_model = "not_a_model"

        # When & Then
        with pytest.raises(ValueError, match="Model must be an instance of MarketDataModel"):
            MarketDataMapper.model_to_entity(invalid_model)

    @pytest.mark.parametrize("open_price,high_price,low_price,close_price", [
        (NEGATIVE_PRICE, TEST_HIGH, TEST_LOW, TEST_CLOSE),
        (TEST_OPEN, NEGATIVE_PRICE, TEST_LOW, TEST_CLOSE),
        (TEST_OPEN, TEST_HIGH, NEGATIVE_PRICE, TEST_CLOSE),
        (TEST_OPEN, TEST_HIGH, TEST_LOW, NEGATIVE_PRICE)
    ])
    def test_model_to_entity_with_negative_prices(
        self, open_price: float, high_price: float, low_price: float, close_price: float
    ) -> None:
        """Test model_to_entity accepts negative prices (for some markets)."""
        # Given
        model = MarketDataModel(
            symbol=TEST_SYMBOL,
            timeframe=Timeframe.MINUTE_5,
            timestamp=TEST_TIMESTAMP,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=TEST_VOLUME
        )

        # When
        result = MarketDataMapper.model_to_entity(model)

        # Then
        assert result.open_price == open_price
        assert result.high_price == high_price
        assert result.low_price == low_price
        assert result.close_price == close_price


class TestMarketDataMapperRoundTrip:
    """Test suite for round-trip conversions between entity and model."""

    def test_entity_to_model_to_entity_round_trip(self, sample_market_data_entity: MarketDataEntity) -> None:
        """Test that entity -> model -> entity preserves all data."""
        # Given
        original_entity = sample_market_data_entity

        # When
        model = MarketDataMapper.entity_to_model(original_entity)
        result_entity = MarketDataMapper.model_to_entity(model)

        # Then
        assert result_entity.symbol == original_entity.symbol
        assert result_entity.timeframe == original_entity.timeframe
        assert result_entity.timestamp == original_entity.timestamp
        assert result_entity.open_price == original_entity.open_price
        assert result_entity.high_price == original_entity.high_price
        assert result_entity.low_price == original_entity.low_price
        assert result_entity.close_price == original_entity.close_price
        assert result_entity.volume == original_entity.volume

    def test_model_to_entity_to_model_round_trip(self, sample_market_data_model: MarketDataModel) -> None:
        """Test that model -> entity -> model preserves all data."""
        # Given
        original_model = sample_market_data_model

        # When
        entity = MarketDataMapper.model_to_entity(original_model)
        result_model = MarketDataMapper.entity_to_model(entity)

        # Then
        assert result_model.symbol == original_model.symbol
        assert result_model.timeframe == original_model.timeframe
        assert result_model.timestamp == original_model.timestamp
        assert result_model.open_price == original_model.open_price
        assert result_model.high_price == original_model.high_price
        assert result_model.low_price == original_model.low_price
        assert result_model.close_price == original_model.close_price
        assert result_model.volume == original_model.volume
