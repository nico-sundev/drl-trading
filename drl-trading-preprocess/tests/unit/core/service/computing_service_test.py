"""Unit tests for the ComputingService class."""

from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeaturesConfig
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)

from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_preprocess.core.service.compute.computing_service import FeatureComputingService


class TestComputingService:
    """Tests for the ComputingService class."""

    @pytest.fixture
    def config_mock(self) -> Mock:
        """Create mock for FeaturesConfig."""
        return Mock(spec=FeaturesConfig)

    @pytest.fixture
    def feature_factory_mock(self) -> Mock:
        """Create mock for FeatureFactoryInterface."""
        return Mock(spec=IFeatureFactory)

    @pytest.fixture
    def feature_manager_service_mock(self) -> Mock:
        """Create mock for FeatureManagerServiceInterface."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def computing_service(self, config_mock, feature_factory_mock, feature_manager_service_mock) -> FeatureComputingService:
        """Create a ComputingService instance with mocked dependencies."""
        return FeatureComputingService(
            feature_manager_service=feature_manager_service_mock
        )

    def test_initialize(self, computing_service, feature_manager_service_mock):
        """Test initialize method."""
        # Given
        feature_manager_service_mock.initialize_features.return_value = None

        # When
        result = computing_service.initialize()

        # Then
        assert result is True
        assert computing_service._initialized is True
        feature_manager_service_mock.initialize_features.assert_called_once_with()

    def test_initialize_exception(self, computing_service, feature_manager_service_mock):
        """Test initialize method with exception."""
        # Given
        feature_manager_service_mock.initialize_features.side_effect = Exception("Test error")

        # When
        result = computing_service.initialize()

        # Then
        assert result is False
        assert computing_service._initialized is False

    def test_compute_batch(self, computing_service, feature_manager_service_mock):
        """Test compute_batch method."""
        # Given
        sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected_result = pd.DataFrame({"Feature1": [10, 20, 30], "Feature2": [40, 50, 60]})
        computing_service._initialized = True
        feature_manager_service_mock.compute_all.return_value = expected_result

        # When
        result = computing_service.compute_batch(sample_df)

        # Then
        feature_manager_service_mock.add.assert_called_once_with(sample_df)
        feature_manager_service_mock.compute_all.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_result)

    def test_compute_batch_not_initialized(self, computing_service, feature_manager_service_mock):
        """Test compute_batch when not initialized."""
        # Given
        sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected_result = pd.DataFrame({"Feature1": [10, 20, 30], "Feature2": [40, 50, 60]})
        computing_service._initialized = False
        feature_manager_service_mock.compute_all.return_value = expected_result
        feature_manager_service_mock.initialize_features.return_value = None

        # When
        result = computing_service.compute_batch(sample_df)

        # Then
        feature_manager_service_mock.initialize_features.assert_called_once_with()
        assert computing_service._initialized is True
        pd.testing.assert_frame_equal(result, expected_result)

    def test_compute_incremental(self, computing_service, feature_manager_service_mock):
        """Test compute_incremental method."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        expected_df = pd.DataFrame({"Feature1": [10], "Feature2": [40]})
        computing_service._initialized = True
        feature_manager_service_mock.compute_latest.return_value = expected_df

        # When
        result = computing_service.compute_incremental(sample_series)

        # Then
        feature_manager_service_mock.add.assert_called_once()
        feature_manager_service_mock.compute_latest.assert_called_once()
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, expected_df.iloc[-1])

    def test_compute_incremental_not_initialized(self, computing_service, feature_manager_service_mock):
        """Test compute_incremental when not initialized."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        computing_service._initialized = False

        # When
        result = computing_service.compute_incremental(sample_series)

        # Then
        assert isinstance(result, pd.Series)
        assert result.empty
        feature_manager_service_mock.add.assert_not_called()
        feature_manager_service_mock.compute_latest.assert_not_called()

    def test_compute_incremental_empty_result(self, computing_service, feature_manager_service_mock):
        """Test compute_incremental with empty result."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        computing_service._initialized = True
        feature_manager_service_mock.compute_latest.return_value = None

        # When
        result = computing_service.compute_incremental(sample_series)

        # Then
        assert isinstance(result, pd.Series)
        assert result.empty
