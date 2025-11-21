"""Test configuration for mapper unit tests."""

import pytest
from datetime import datetime, timezone
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_common.model.timeframe import Timeframe


@pytest.fixture
def sample_market_data_entity() -> MarketDataEntity:
    """Create a sample MarketDataEntity for testing."""
    entity = MarketDataEntity()
    entity.symbol = "AAPL"
    entity.timeframe = "1h"
    entity.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    entity.open_price = 150.25
    entity.high_price = 152.75
    entity.low_price = 149.50
    entity.close_price = 151.80
    entity.volume = 1000000
    return entity


@pytest.fixture
def sample_market_data_model() -> MarketDataModel:
    """Create a sample MarketDataModel for testing."""
    return MarketDataModel(
        symbol="AAPL",
        timeframe=Timeframe.HOUR_1,
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        open_price=150.25,
        high_price=152.75,
        low_price=149.50,
        close_price=151.80,
        volume=1000000
    )


# Test data fixtures
@pytest.fixture
def test_symbol() -> str:
    """Test symbol constant."""
    return "TSLA"


@pytest.fixture
def test_timeframe() -> str:
    """Test timeframe constant."""
    return "5m"


@pytest.fixture
def test_timestamp():
    """Test timestamp constant."""
    return datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def test_open() -> float:
    """Test open price constant."""
    return 200.50


@pytest.fixture
def test_high() -> float:
    """Test high price constant."""
    return 205.75


@pytest.fixture
def test_low() -> float:
    """Test low price constant."""
    return 198.25


@pytest.fixture
def test_close() -> float:
    """Test close price constant."""
    return 203.40


@pytest.fixture
def test_volume() -> int:
    """Test volume constant."""
    return 2500000


@pytest.fixture
def zero_volume() -> int:
    """Zero volume constant."""
    return 0


@pytest.fixture
def negative_price() -> float:
    """Negative price constant."""
    return -1.0


@pytest.fixture
def very_large_volume() -> int:
    """Very large volume constant."""
    return 999999999999


@pytest.fixture
def test_market_data_entity(test_symbol, test_timeframe, test_timestamp, test_open, test_high, test_low, test_close, test_volume) -> MarketDataEntity:
    """Create a test MarketDataEntity with standard test data."""
    entity = MarketDataEntity()
    entity.symbol = test_symbol
    entity.timeframe = test_timeframe
    entity.timestamp = test_timestamp
    entity.open_price = test_open
    entity.high_price = test_high
    entity.low_price = test_low
    entity.close_price = test_close
    entity.volume = test_volume
    return entity


@pytest.fixture
def test_market_data_model(test_symbol, test_timestamp, test_open, test_high, test_low, test_close, test_volume) -> MarketDataModel:
    """Create a test MarketDataModel with standard test data."""
    return MarketDataModel(
        symbol=test_symbol,
        timeframe=Timeframe.MINUTE_5,  # Assuming 5m maps to MINUTE_5
        timestamp=test_timestamp,
        open_price=test_open,
        high_price=test_high,
        low_price=test_low,
        close_price=test_close,
        volume=test_volume
    )
