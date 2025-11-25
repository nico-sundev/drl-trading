"""Unit tests for MarketDataMapper.

Tests the mapping functionality between MarketDataEntity and MarketDataModel,
including error handling and edge cases.
"""

import pytest
from datetime import datetime, timezone
from drl_trading_adapter.adapter.database.mapper.market_data_mapper import MarketDataMapper
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_core.core.model.market_data_model import MarketDataModel

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

    def test_entity_to_model_with_missing_required_fields(self, test_market_data_entity: MarketDataEntity) -> None:
        """Test entity_to_model with missing required fields raises ValueError."""
        # Given
        entity = test_market_data_entity
        entity.open_price = None  # Intentionally set a required field to None

        # When & Then
        with pytest.raises(ValueError, match="Entity has None values in required fields"):
            MarketDataMapper.entity_to_model(entity)

    @pytest.mark.parametrize("symbol,timeframe,volume", [
        ("TSLA", "5m", 0),
        ("ABC", "5m", 2500000),  # Different but valid symbol
        ("TSLA", "1h", 2500000),  # Different but valid timeframe
        ("TSLA", "5m", 999999999999)
    ])
    def test_entity_to_model_edge_cases(self, symbol: str, timeframe: str, volume: int) -> None:
        """Test entity_to_model with edge case values."""
        # Given
        entity = MarketDataEntity()
        entity.symbol = symbol
        entity.timeframe = timeframe
        entity.timestamp = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        entity.open_price = 200.50
        entity.high_price = 205.75
        entity.low_price = 198.25
        entity.close_price = 203.40
        entity.volume = volume

        # When
        result = MarketDataMapper.entity_to_model(entity)

        # Then
        assert result.symbol == symbol
        assert result.timeframe.value == timeframe  # Compare with the enum value
        assert result.volume == volume

    def test_entity_to_model_with_empty_symbol_succeeds(self, test_market_data_entity: MarketDataEntity) -> None:
        """Test entity_to_model with empty symbol succeeds (empty string is valid)."""
        # Given
        entity = test_market_data_entity
        entity.symbol = ""  # Empty symbol

        # When
        result = MarketDataMapper.entity_to_model(entity)

        # Then
        assert result.symbol == ""

    def test_entity_to_model_with_invalid_timeframe_raises_error(self, test_market_data_entity: MarketDataEntity) -> None:
        """Test entity_to_model with invalid timeframe raises appropriate error."""
        # Given
        entity = test_market_data_entity
        entity.timeframe = ""  # Invalid timeframe

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
        (-1.0, 205.75, 198.25, 203.40),
        (200.50, -1.0, 198.25, 203.40),
        (200.50, 205.75, -1.0, 203.40),
        (200.50, 205.75, 198.25, -1.0)
    ])
    def test_model_to_entity_with_negative_prices(
        self, open_price: float, high_price: float, low_price: float, close_price: float, test_market_data_model: MarketDataModel
    ) -> None:
        """Test model_to_entity accepts negative prices (for some markets)."""
        # Given
        model = test_market_data_model
        model.open_price = open_price
        model.high_price = high_price
        model.low_price = low_price
        model.close_price = close_price

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
