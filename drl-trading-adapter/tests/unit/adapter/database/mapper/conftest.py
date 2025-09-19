"""Test configuration for mapper unit tests."""

import pytest
from datetime import datetime
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_common.model.timeframe import Timeframe


@pytest.fixture
def sample_market_data_entity() -> MarketDataEntity:
    """Create a sample MarketDataEntity for testing."""
    entity = MarketDataEntity()
    entity.symbol = "AAPL"
    entity.timeframe = "1h"
    entity.timestamp = datetime(2024, 1, 15, 10, 0, 0)
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
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        open_price=150.25,
        high_price=152.75,
        low_price=149.50,
        close_price=151.80,
        volume=1000000
    )


# Constants for test data
TEST_SYMBOL = "TSLA"
TEST_TIMEFRAME = "5m"
TEST_TIMESTAMP = datetime(2024, 1, 15, 14, 30, 0)
TEST_OPEN = 200.50
TEST_HIGH = 205.75
TEST_LOW = 198.25
TEST_CLOSE = 203.40
TEST_VOLUME = 2500000

# Edge case values
ZERO_VOLUME = 0
NEGATIVE_PRICE = -1.0
VERY_LARGE_VOLUME = 999999999999
