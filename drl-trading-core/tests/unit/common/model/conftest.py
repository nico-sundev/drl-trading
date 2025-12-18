"""Fixtures for common model tests."""
from typing import Optional
from unittest.mock import MagicMock

import pytest
from drl_trading_core.core.port.base_feature import BaseFeature
from drl_trading_core.core.model.feature.feature_metadata import FeatureMetadata
from drl_trading_common.core.model.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_core.core.port.technical_indicator_service_port import ITechnicalIndicatorServicePort
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.core.model.timeframe import Timeframe
from pandas import DataFrame


class MockFeature(BaseFeature):
    """Mock feature implementation for testing."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: ITechnicalIndicatorServicePort,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
        feature_name: str = "mock_feature"
    ) -> None:
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = feature_name

    def _call_indicator_backend(self, method_call) -> Optional[DataFrame]:
        """Mock implementation."""
        return None

    def update(self, df: DataFrame) -> None:
        """Mock implementation of add method."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Mock implementation of compute_latest method."""
        return DataFrame()

    def compute_all(self) -> Optional[DataFrame]:
        """Mock implementation of compute_all method."""
        return DataFrame()

    def _get_sub_features_names(self) -> list[str]:
        """Mock implementation of get_sub_features_names method."""
        return [f"{self._feature_name}{self.postfix}"]

    def _get_feature_name(self) -> str:
        """Mock implementation of get_feature_name method."""
        return self._feature_name

    def _get_config_to_string(self) -> Optional[str]:
        """Mock implementation of get_config_to_string method."""
        return "mock_config" if self.config else None


@pytest.fixture
def mock_feature() -> MockFeature:
    """Create a mock feature for testing."""
    mock_indicator_service = MagicMock(spec=ITechnicalIndicatorServicePort)
    mock_dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe=Timeframe.HOUR_1)

    return MockFeature(
        dataset_id=mock_dataset_id,
        indicator_service=mock_indicator_service,
        feature_name="test_feature"
    )


@pytest.fixture
def mock_feature_metadata() -> FeatureMetadata:
    """Create a mock feature metadata for testing."""
    return FeatureMetadata(
        config="mock_config",
        dataset_id=DatasetIdentifier("EURUSD", Timeframe.HOUR_1),
        feature_role="observation_space",
        feature_name="test_feature",
        sub_feature_names=["test_feature"]
    )


@pytest.fixture
def mock_features_list(mock_feature: MockFeature) -> list[BaseFeature]:
    """Create a list of mock features for testing."""
    return [mock_feature]


@pytest.fixture
def mock_timeframe() -> Timeframe:
    """Create a mock timeframe for testing."""
    return Timeframe.HOUR_1


@pytest.fixture
def mock_dataset_identifier() -> DatasetIdentifier:
    """Create a mock dataset identifier for testing."""
    return DatasetIdentifier(symbol="EURUSD", timeframe=Timeframe.HOUR_1)
