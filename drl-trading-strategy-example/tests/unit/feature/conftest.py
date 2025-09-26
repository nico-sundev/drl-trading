from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.model.dataset_identifier import DatasetIdentifier


class MockMacdFeature(BaseFeature):
    """Mock MACD feature implementation for testing."""

    def __init__(
        self,
        config: BaseParameterSetConfig,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        postfix: str = "",
    ) -> None:
        super().__init__(config, dataset_id, indicator_service, postfix)
        self.df_source = None
        self.feature_name = f"macd{postfix}"

    def update(self, df: pd.DataFrame) -> None:
        """Add new data to the feature."""
        self.df_source = df

    def compute_all(self) -> Optional[pd.DataFrame]:
        """Compute all feature values."""
        if self.df_source is None:
            return None

        # Return mock MACD data with proper structure
        result_df = pd.DataFrame(index=self.df_source.index)
        result_df[f"macd{self.postfix}"] = 0.1
        result_df[f"macd_signal{self.postfix}"] = 0.05
        result_df[f"macd_histogram{self.postfix}"] = 0.05
        return result_df

    def compute_latest(self) -> Optional[pd.DataFrame]:
        """Compute latest feature values."""
        if self.df_source is None or self.df_source.empty:
            return None

        # Return mock latest MACD data
        latest_index = [self.df_source.index[-1]]
        result_df = pd.DataFrame(index=latest_index)
        result_df[f"macd{self.postfix}"] = 0.1
        result_df[f"macd_signal{self.postfix}"] = 0.05
        result_df[f"macd_histogram{self.postfix}"] = 0.05
        return result_df

    def get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features."""
        return [
            f"macd{self.postfix}",
            f"macd_signal{self.postfix}",
            f"macd_histogram{self.postfix}"
        ]

    def get_feature_name(self) -> str:
        """Get the feature name."""
        return "macd"

    def get_config_to_string(self) -> str:
        return "A1b2c3"


class MockRsiFeature(BaseFeature):
    """Mock RSI feature implementation for testing."""

    def __init__(
        self,
        config: BaseParameterSetConfig,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorFacade,
        postfix: str = "",
    ) -> None:
        super().__init__(config, dataset_id, indicator_service, postfix)
        self.df_source = None
        self.feature_name = f"rsi{postfix}"

    def update(self, df: pd.DataFrame) -> None:
        """Add new data to the feature."""
        self.df_source = df

    def compute_all(self) -> Optional[pd.DataFrame]:
        """Compute all feature values."""
        if self.df_source is None:
            return None

        # Return mock RSI data with proper structure
        result_df = pd.DataFrame(index=self.df_source.index)
        result_df[f"rsi{self.postfix}"] = 50.0  # Neutral RSI value
        return result_df

    def compute_latest(self) -> Optional[pd.DataFrame]:
        """Compute latest feature values."""
        if self.df_source is None or self.df_source.empty:
            return None

        # Return mock latest RSI data
        latest_index = [self.df_source.index[-1]]
        result_df = pd.DataFrame(index=latest_index)
        result_df[f"rsi{self.postfix}"] = 50.0
        return result_df

    def get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features."""
        return [f"rsi{self.postfix}"]

    def get_feature_name(self) -> str:
        """Get the feature name."""
        return "rsi"

    def get_config_to_string(self) -> str:
        return "A1b2c3"


@pytest.fixture
def mock_macd_feature_class():
    """Fixture providing the MockMacdFeature class."""
    return MockMacdFeature


@pytest.fixture
def mock_rsi_feature_class():
    """Fixture providing the MockRsiFeature class."""
    return MockRsiFeature


@pytest.fixture
def mock_indicator_service():
    """Fixture providing a mock indicator service."""
    return MagicMock(spec=ITechnicalIndicatorFacade)


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=dates)
