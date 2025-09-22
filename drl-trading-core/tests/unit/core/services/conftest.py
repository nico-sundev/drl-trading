"""Fixtures specific to feature processing tests."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.timeframe import Timeframe
from pandas import DataFrame

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet


@pytest.fixture
def mock_feature_definition() -> FeatureDefinition:
    """Create a mock feature definition for feature tests."""
    mock_param_set = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set.enabled = True
    mock_param_set.hash_id = lambda: "abc123hash"
    mock_param_set.to_string = lambda: "7_14"
    mock_param_set.name = "test_params"

    feature_def = MagicMock(spec=FeatureDefinition)
    feature_def.name = "MockFeature"
    feature_def.enabled = True
    feature_def.parsed_parameter_sets = [mock_param_set]
    return feature_def


@pytest.fixture
def mock_param_set() -> BaseParameterSetConfig:
    """Create a mock parameter set for feature tests."""
    mock_param_set = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set.enabled = True
    mock_param_set.hash_id = lambda: "abc123hash"
    mock_param_set.to_string = lambda: "7_14"
    mock_param_set.name = "test_params"
    return mock_param_set


@pytest.fixture
def mock_param_set_drop_time() -> BaseParameterSetConfig:
    """Create a mock parameter set that will cause Time index to be dropped."""
    mock_param_set = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set.enabled = True
    mock_param_set.hash_id = lambda: "drop_time_hash"
    mock_param_set.to_string = lambda: "drop_time"
    mock_param_set.name = "drop_time_params"
    return mock_param_set


@pytest.fixture
def feature_test_asset_df() -> DataFrame:
    """Create a standardized DataFrame for feature testing."""
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=10, freq="h"))
    data = {
        "Open": [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.35, 1.5],
        "High": [1.1, 1.2, 1.15, 1.3, 1.25, 1.4, 1.35, 1.5, 1.45, 1.6],
        "Low": [0.9, 1.0, 0.95, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4],
        "Close": [1.05, 1.15, 1.1, 1.25, 1.2, 1.35, 1.3, 1.45, 1.4, 1.55],
        "Volume": [
            1000.0,
            1100.0,
            950.0,
            1200.0,
            1050.0,
            1300.0,
            1150.0,
            1400.0,
            1250.0,
            1500.0,
        ],
    }
    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def feature_test_asset_data(feature_test_asset_df) -> AssetPriceDataSet:
    """Create a standardized asset price dataset for feature testing."""
    return AssetPriceDataSet(
        timeframe=Timeframe.HOUR_1,
        base_dataset=True,
        asset_price_dataset=feature_test_asset_df,
    )


@pytest.fixture
def feature_test_symbol() -> str:
    """Standard symbol for feature tests."""
    return "EURUSD"
