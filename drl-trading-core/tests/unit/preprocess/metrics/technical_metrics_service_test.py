from unittest.mock import Mock, patch

import pandas as pd
import pytest

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsService,
    TechnicalMetricsServiceFactory,
)


@pytest.fixture
def sample_price_data():
    """Create sample OHLC data with price movements for testing."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    data = {
        "Open": [100.0, 102.0, 104.0, 103.0, 105.0] * 4,
        "High": [105.0, 107.0, 106.0, 105.0, 110.0] * 4,
        "Low": [98.0, 100.0, 102.0, 101.0, 103.0] * 4,
        "Close": [102.0, 104.0, 103.0, 105.0, 108.0] * 4,
        "Volume": [1000] * 20,
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def mock_asset_data(sample_price_data):
    """Create a mock AssetPriceDataSet with the sample data."""
    mock_asset = Mock(spec=AssetPriceDataSet)
    mock_asset.timeframe = "D1"
    mock_asset.asset_price_dataset = sample_price_data
    mock_asset.base_dataset = True
    return mock_asset


@pytest.fixture
def metrics_service(mock_asset_data):
    """Create a TechnicalMetricsService instance for testing."""
    return TechnicalMetricsService(mock_asset_data)


def test_get_atr(sample_price_data, mock_asset_data):
    """Test that ATR is calculated correctly with the specified period."""
    # Given
    # Using mock_asset_data fixture with sample price data

    # When
    # Create service and request ATR with period=5
    service = TechnicalMetricsService(mock_asset_data)
    atr_df = service.get_atr(period=5)

    # Then
    # Verify ATR calculation exists and has correct format
    assert atr_df is not None
    assert "ATR" in atr_df.columns
    assert atr_df.index.name == "Time"
    assert isinstance(atr_df.index, pd.DatetimeIndex)

    # ATR should be positive since it measures volatility
    assert (atr_df["ATR"].dropna() > 0).all()

    # Check that first period-1 rows have NaN (due to rolling window)
    assert atr_df["ATR"].iloc[:4].isna().all()

    # Check that remaining values are not NaN
    assert not atr_df["ATR"].iloc[4:].isna().any()


def test_get_atr_from_cache(metrics_service):
    """Test that ATR is returned from cache on subsequent calls."""
    # Given
    # Using metrics_service fixture

    # When
    # Call get_atr twice with the same period
    with patch.object(
        TechnicalMetricsService, "_calculate_atr", wraps=metrics_service._calculate_atr
    ) as mock_calc:
        first_call = metrics_service.get_atr(period=14)
        second_call = metrics_service.get_atr(period=14)

    # Then
    # Verify the calculation method was called only once
    mock_calc.assert_called_once()
    # Verify both calls return the same DataFrame object
    assert first_call is second_call


@pytest.fixture
def h1_mock_asset_data():
    """Create a mock AssetPriceDataSet with H1 timeframe."""
    mock_asset = Mock(spec=AssetPriceDataSet)
    mock_asset.timeframe = "H1"
    dates = pd.date_range(start="2023-01-01 00:00:00", periods=20, freq="H")
    data = {"Open": [100.0], "High": [105.0], "Low": [98.0], "Close": [102.0]}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Time"
    mock_asset.asset_price_dataset = df
    mock_asset.base_dataset = False
    return mock_asset


def test_factory_create(h1_mock_asset_data):
    """Test that factory creates a service instance with correct timeframe."""
    # Given
    # Using h1_mock_asset_data fixture

    # When
    # Create service using factory
    service = TechnicalMetricsServiceFactory.create(h1_mock_asset_data)

    # Then
    # Verify service has correct timeframe
    assert service.timeframe == "H1"
