from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_framework.common.config.feature_config_collection import (
    BollbandsConfig,
)
from drl_trading_framework.preprocess.feature.collection.bollbands_feature import (
    BollbandsFeature,
)
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)


@pytest.fixture
def sample_data() -> DataFrame:
    """Create sample price data for testing."""
    # Given
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    data = {
        "Open": np.random.normal(100, 5, 50),
        "High": np.random.normal(105, 5, 50),
        "Low": np.random.normal(95, 5, 50),
        "Close": np.random.normal(100, 5, 50),
        "Volume": [1000] * 50,
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def bollbands_config() -> BollbandsConfig:
    """Create a BollbandsConfig for testing."""
    return BollbandsConfig(length=20, std_dev=2.0, enabled=True, name="default")


@pytest.fixture
def metrics_service() -> Mock:
    """Create a mock metrics service that returns ATR values."""
    mock_metrics = Mock(spec=TechnicalMetricsServiceInterface)
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    atr_values = pd.DataFrame({"ATR": np.random.normal(5, 1, 50)}, index=dates)
    atr_values.index.name = "Time"
    mock_metrics.get_atr.return_value = atr_values
    return mock_metrics


def test_compute_without_metrics(sample_data, bollbands_config):
    """Test computing Bollinger Bands without metrics service."""
    # Given
    feature = BollbandsFeature(source=sample_data, config=bollbands_config)

    # When
    result_df = feature.compute()

    # Then
    assert "bb_upper" in result_df.columns
    assert "bb_middle" in result_df.columns
    assert "bb_lower" in result_df.columns

    # ATR-adjusted bands should not be present
    assert "bb_atr_upper" not in result_df.columns
    assert "bb_atr_lower" not in result_df.columns
    assert "bb_atr_width" not in result_df.columns

    # Feature names should match calculated columns
    assert feature.get_sub_features_names() == ["bb_upper", "bb_middle", "bb_lower"]


def test_compute_with_metrics(sample_data, bollbands_config, metrics_service):
    """Test computing Bollinger Bands with metrics service for ATR-adjusted bands."""
    # Given
    feature = BollbandsFeature(
        source=sample_data, config=bollbands_config, metrics_service=metrics_service
    )

    # When
    result_df = feature.compute()

    # Then
    # Standard bands should be present
    assert "bb_upper" in result_df.columns
    assert "bb_middle" in result_df.columns
    assert "bb_lower" in result_df.columns

    # ATR-adjusted bands should be present
    assert "bb_atr_upper" in result_df.columns
    assert "bb_atr_lower" in result_df.columns
    assert "bb_atr_width" in result_df.columns

    # Metrics service should have been called with correct period
    metrics_service.get_atr.assert_called_once_with(period=20)

    # Feature names should include ATR-related features
    expected_features = [
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_atr_upper",
        "bb_atr_lower",
        "bb_atr_width",
    ]
    assert feature.get_sub_features_names() == expected_features
