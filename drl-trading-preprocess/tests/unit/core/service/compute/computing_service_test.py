"""Unit tests for the ComputingService class."""

import logging
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeaturesConfig
from drl_trading_common.model.timeframe import Timeframe

from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_preprocess.core.service.compute.computing_service import FeatureComputingService


class TestComputingService:
    """Tests for the ComputingService class."""

    @pytest.fixture
    def feature_manager_service_mock(self) -> Mock:
        """Create mock for FeatureManager."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def computing_service(self, feature_manager_service_mock: Mock) -> FeatureComputingService:
        """Create a ComputingService instance with mocked dependencies."""
        return FeatureComputingService(feature_manager=feature_manager_service_mock)

    @pytest.fixture
    def features_config_mock(self) -> Mock:
        """Create mock for FeaturesConfig."""
        config = Mock(spec=FeaturesConfig)
        config.dataset_definitions = {"EURUSD": [Timeframe.MINUTE_5]}
        config.feature_definitions = []
        return config

    def test_compute_batch(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, features_config_mock: Mock) -> None:
        """Test compute_batch method with realistic feature structure."""
        # Given
        sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Realistic result: features should have event_timestamp column (from feature computation)
        timestamps = pd.date_range("2024-01-01", periods=3, freq="1h")
        expected_result = pd.DataFrame({
            "event_timestamp": timestamps,
            "Feature1": [10, 20, 30],
            "Feature2": [40, 50, 60]
        })
        feature_manager_service_mock.compute_all.return_value = expected_result

        # When
        result = computing_service.compute_batch(sample_df, features_config_mock)

        # Then
        feature_manager_service_mock.request_features_update.assert_called_once_with(sample_df, features_config_mock)
        feature_manager_service_mock.compute_all.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_result)
        # Verify only ONE event_timestamp column exists
        assert list(result.columns).count("event_timestamp") == 1, "Should have exactly one event_timestamp column"

    def test_compute_batch_empty_result(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, features_config_mock: Mock) -> None:
        """Test compute_batch with empty result."""
        # Given
        sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        feature_manager_service_mock.compute_all.return_value = None

        # When
        result = computing_service.compute_batch(sample_df, features_config_mock)

        # Then
        feature_manager_service_mock.request_features_update.assert_called_once_with(sample_df, features_config_mock)
        feature_manager_service_mock.compute_all.assert_called_once()
        assert result.empty

    def test_compute_incremental(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, features_config_mock: Mock) -> None:
        """Test compute_incremental method."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        expected_df = pd.DataFrame({"Feature1": [10], "Feature2": [40]})
        feature_manager_service_mock.compute_latest.return_value = expected_df

        # When
        result = computing_service.compute_incremental(sample_series, features_config_mock)

        # Then
        # Should convert series to dataframe for features update
        expected_df_call = pd.DataFrame([sample_series])
        feature_manager_service_mock.request_features_update.assert_called_once()
        call_args = feature_manager_service_mock.request_features_update.call_args
        pd.testing.assert_frame_equal(call_args[0][0], expected_df_call)
        assert call_args[0][1] == features_config_mock

        feature_manager_service_mock.compute_latest.assert_called_once()
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, expected_df.iloc[-1])

    def test_compute_incremental_empty_result(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, features_config_mock: Mock) -> None:
        """Test compute_incremental with empty result."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        feature_manager_service_mock.compute_latest.return_value = None

        # When
        result = computing_service.compute_incremental(sample_series, features_config_mock)

        # Then
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_compute_incremental_empty_dataframe_result(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, features_config_mock: Mock) -> None:
        """Test compute_incremental with empty DataFrame result."""
        # Given
        sample_series = pd.Series({"A": 1, "B": 4})
        feature_manager_service_mock.compute_latest.return_value = pd.DataFrame()

        # When
        result = computing_service.compute_incremental(sample_series, features_config_mock)

        # Then
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_check_catchup_status_mixed_results(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock) -> None:
        """Test check_catchup_status method with mixed results."""
        # Given
        reference_time = datetime.now()
        mock_features = {
            "feature1": Mock(),
            "feature2": Mock()
        }
        mock_features["feature1"].are_features_caught_up.return_value = True
        mock_features["feature2"].are_features_caught_up.return_value = False

        feature_manager_service_mock._features = mock_features

        # When
        result = computing_service.check_catchup_status(reference_time)

        # Then
        assert result.total_features == 2
        assert len(result.caught_up_features) == 1
        assert len(result.not_caught_up_features) == 1
        assert result.catch_up_percentage == 50.0
        assert not result.all_caught_up
        assert result.has_features_needing_warmup()

    def test_check_catchup_status_no_features_attribute(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock) -> None:
        """Test check_catchup_status when feature_manager has no _features attribute."""
        # Given
        reference_time = datetime.now()
        # Remove _features attribute to simulate missing attribute
        if hasattr(feature_manager_service_mock, '_features'):
            delattr(feature_manager_service_mock, '_features')

        # When
        result = computing_service.check_catchup_status(reference_time)

        # Then
        assert result.total_features == 0
        assert len(result.caught_up_features) == 0
        assert len(result.not_caught_up_features) == 0
        assert result.catch_up_percentage == 0.0
        assert not result.all_caught_up
        assert not result.has_features_needing_warmup()
        assert result.reference_time == reference_time

    def test_check_catchup_status_empty_features(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock) -> None:
        """Test check_catchup_status when _features is empty."""
        # Given
        reference_time = datetime.now()
        feature_manager_service_mock._features = {}

        # When
        result = computing_service.check_catchup_status(reference_time)

        # Then
        assert result.total_features == 0
        assert len(result.caught_up_features) == 0
        assert len(result.not_caught_up_features) == 0
        assert result.catch_up_percentage == 0.0
        assert not result.all_caught_up
        assert not result.has_features_needing_warmup()
        assert result.reference_time == reference_time

    def test_check_catchup_status_all_caught_up(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock) -> None:
        """Test check_catchup_status when all features are caught up."""
        # Given
        reference_time = datetime.now()
        mock_features = {
            "feature1": Mock(),
            "feature2": Mock(),
            "feature3": Mock()
        }
        mock_features["feature1"].are_features_caught_up.return_value = True
        mock_features["feature2"].are_features_caught_up.return_value = True
        mock_features["feature3"].are_features_caught_up.return_value = True

        feature_manager_service_mock._features = mock_features

        # When
        result = computing_service.check_catchup_status(reference_time)

        # Then
        assert result.total_features == 3
        assert len(result.caught_up_features) == 3
        assert len(result.not_caught_up_features) == 0
        assert result.catch_up_percentage == 100.0
        assert result.all_caught_up
        assert not result.has_features_needing_warmup()
        assert result.reference_time == reference_time

    def test_check_catchup_status_exception_handling(self, computing_service: FeatureComputingService, feature_manager_service_mock: Mock, caplog: pytest.LogCaptureFixture) -> None:
        """Test check_catchup_status when feature.are_features_caught_up raises exception."""
        # Given
        reference_time = datetime.now()
        mock_feature_good = Mock()
        mock_feature_bad = Mock()
        mock_feature_good.are_features_caught_up.return_value = True
        mock_feature_bad.are_features_caught_up.side_effect = RuntimeError("Feature computation error")

        mock_features = {
            "good_feature": mock_feature_good,
            "bad_feature": mock_feature_bad
        }
        feature_manager_service_mock._features = mock_features

        with caplog.at_level(logging.ERROR):
            # When
            result = computing_service.check_catchup_status(reference_time)

        # Then
        assert result.total_features == 2
        assert len(result.caught_up_features) == 1
        assert len(result.not_caught_up_features) == 1
        assert "good_feature" in result.caught_up_features
        assert "bad_feature" in result.not_caught_up_features
        assert result.catch_up_percentage == 50.0
        assert not result.all_caught_up
        assert result.has_features_needing_warmup()

        # Check error logging
        assert "Error checking catch-up status for feature bad_feature" in caplog.text
        assert "Feature computation error" in caplog.text


class TestComputingServiceFeatureDeduplication:
    """Test cases for event_timestamp deduplication when computing multiple features."""

    @pytest.fixture
    def feature_manager_service_mock(self) -> Mock:
        """Create mock for FeatureManager."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def computing_service(self, feature_manager_service_mock: Mock) -> FeatureComputingService:
        """Create a ComputingService instance with mocked dependencies."""
        return FeatureComputingService(feature_manager=feature_manager_service_mock)

    @pytest.fixture
    def features_config_mock(self) -> Mock:
        """Create mock for FeaturesConfig."""
        config = Mock(spec=FeaturesConfig)
        config.dataset_definitions = {"EURUSD": [Timeframe.MINUTE_5]}
        config.feature_definitions = []
        return config

    def test_compute_batch_deduplicates_event_timestamp(
        self,
        computing_service: FeatureComputingService,
        feature_manager_service_mock: Mock,
        features_config_mock: Mock
    ) -> None:
        """Test that compute_batch handles duplicate event_timestamp columns correctly.

        This simulates the real-world scenario where FeatureManager._combine_dataframes_efficiently
        is called and each feature's DataFrame includes an event_timestamp column.
        The FeatureManager should deduplicate these, so we test that the final result
        from compute_batch has exactly one event_timestamp column.
        """
        # Given: Market data to compute features on
        sample_df = pd.DataFrame({"close": [1.1, 1.2, 1.3], "volume": [100, 200, 300]})

        # Simulate FeatureManager returning already-deduplicated features
        # (this is what FeatureManager._combine_dataframes_efficiently should do)
        timestamps = pd.date_range("2024-01-01", periods=3, freq="1h")
        result_from_feature_manager = pd.DataFrame({
            "event_timestamp": timestamps,
            "rsi_14": [30.5, 45.2, 67.8],
            "sma_20": [1.0850, 1.0855, 1.0860],
            "bb_upper": [1.0870, 1.0875, 1.0880]
        })
        feature_manager_service_mock.compute_all.return_value = result_from_feature_manager

        # When: Computing features
        result = computing_service.compute_batch(sample_df, features_config_mock)

        # Then: Should have exactly ONE event_timestamp column
        assert "event_timestamp" in result.columns, "Result must contain event_timestamp"
        event_timestamp_count = list(result.columns).count("event_timestamp")
        assert event_timestamp_count == 1, (
            f"Expected exactly 1 event_timestamp column, found {event_timestamp_count}. "
            f"Columns: {list(result.columns)}"
        )

        # Then: Should have all feature columns
        assert "rsi_14" in result.columns
        assert "sma_20" in result.columns
        assert "bb_upper" in result.columns

        # Then: Should have correct number of rows
        assert len(result) == 3

    def test_compute_incremental_handles_event_timestamp(
        self,
        computing_service: FeatureComputingService,
        feature_manager_service_mock: Mock,
        features_config_mock: Mock
    ) -> None:
        """Test compute_incremental with event_timestamp in result."""
        # Given: Single data point
        sample_series = pd.Series({"close": 1.1, "volume": 100})

        # Simulate FeatureManager returning features with event_timestamp
        timestamps = pd.date_range("2024-01-01", periods=1, freq="1h")
        expected_df = pd.DataFrame({
            "event_timestamp": timestamps,
            "rsi_14": [45.2],
            "sma_20": [1.0855]
        })
        feature_manager_service_mock.compute_latest.return_value = expected_df

        # When: Computing incremental
        result = computing_service.compute_incremental(sample_series, features_config_mock)

        # Then: Should return Series with all columns including event_timestamp
        assert isinstance(result, pd.Series)
        assert "event_timestamp" in result.index
        assert "rsi_14" in result.index
        assert "sma_20" in result.index
