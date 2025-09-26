"""
Unit tests for the PreprocessService.

Tests the main preprocessing service orchestration functionality,
including dynamic feature definitions and request processing.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from pandas import DataFrame

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.feature_computation_request import FeatureComputationRequest
from drl_trading_preprocess.core.model.feature_computation_response import (
    FeatureExistenceCheckResult,
)
from drl_trading_preprocess.core.service.preprocess_service import PreprocessService


class TestPreprocessService:
    """Test cases for PreprocessService."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for PreprocessService."""
        return {
            'market_data_resampler': Mock(),
            'feature_computer': Mock(),
            'feature_store_port': Mock(),
            'feature_existence_checker': Mock(),
        }

    @pytest.fixture
    def preprocess_service(self, mock_dependencies):
        """Create PreprocessService instance with mocked dependencies."""
        return PreprocessService(
            market_data_resampler=mock_dependencies['market_data_resampler'],
            feature_computer=mock_dependencies['feature_computer'],
            feature_store_port=mock_dependencies['feature_store_port'],
            feature_existence_checker=mock_dependencies['feature_existence_checker'],
        )

    @pytest.fixture
    def sample_feature_request(self):
        """Create a sample feature computation request."""
        feature_def = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[14],
            parameter_sets=[{"period": 14}],
        )

        return FeatureComputationRequest(
            symbol="BTCUSDT",
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15],
            feature_definitions=[feature_def],
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            request_id="test_request_123",
            force_recompute=False,
            incremental_mode=True,
        )

    def test_process_feature_computation_request_success(
        self, preprocess_service, mock_dependencies, sample_feature_request
    ):
        """Test successful feature computation request processing."""
        # Given
        # Mock validation passes
        mock_dependencies['feature_computer'].validate_feature_definitions.return_value = {
            "rsi": True
        }

        # Mock no existing features (force computation)
        sample_feature_request.skip_existing_features = False

        # Mock resampling response
        mock_resampling_response = Mock()
        mock_resampling_response.resampled_data = {
            Timeframe.MINUTE_5: [Mock()],  # Mock market data
            Timeframe.MINUTE_15: [Mock()],
        }
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = mock_resampling_response

        # Mock feature computation
        mock_features_df = DataFrame({'rsi_14': [30.0, 35.0, 40.0]})
        mock_dependencies['feature_computer'].compute_features_for_request.return_value = mock_features_df

        # When
        response = preprocess_service.process_feature_computation_request(sample_feature_request)

        # Then
        assert response.success is True
        assert response.request_id == "test_request_123"
        assert response.symbol == "BTCUSDT"
        assert response.stats.features_computed > 0

        # Verify service interactions
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.assert_called_once()
        mock_dependencies['feature_computer'].compute_features_for_request.assert_called()
        mock_dependencies['feature_store_port'].store_computed_features_offline.assert_called()

    def test_process_feature_computation_request_skip_existing_features(
        self, preprocess_service, mock_dependencies, sample_feature_request
    ):
        """Test processing when features already exist and should be skipped."""
        # Given
        # Mock validation passes
        mock_dependencies['feature_computer'].validate_feature_definitions.return_value = {
            "rsi": True
        }

        # Mock existing features check - all features exist
        existing_check_result = FeatureExistenceCheckResult(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            existing_features=["rsi"],
            missing_features=[],
        )
        mock_dependencies['feature_existence_checker'].check_feature_existence.return_value = existing_check_result

        sample_feature_request.skip_existing_features = True
        sample_feature_request.force_recompute = False

        # When
        response = preprocess_service.process_feature_computation_request(sample_feature_request)

        # Then
        assert response.success is True
        assert response.stats.features_computed == 0  # No computation needed
        assert response.stats.features_skipped > 0

        # Verify resampling and computation were not called
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.assert_not_called()
        mock_dependencies['feature_computer'].compute_features_for_request.assert_not_called()

    def test_process_feature_computation_request_validation_failure(
        self, preprocess_service, mock_dependencies, sample_feature_request
    ):
        """Test processing when feature validation fails."""
        # Given
        # Mock validation fails
        mock_dependencies['feature_computer'].validate_feature_definitions.return_value = {
            "rsi": False  # Invalid feature
        }

        # When
        response = preprocess_service.process_feature_computation_request(sample_feature_request)

        # Then
        assert response.success is False
        assert "Invalid feature definitions" in response.error_message
        assert response.stats.features_failed > 0

        # Verify no downstream processing occurred
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.assert_not_called()

    def test_process_feature_computation_request_resampling_failure(
        self, preprocess_service, mock_dependencies, sample_feature_request
    ):
        """Test processing when market data resampling fails."""
        # Given
        # Mock validation passes
        mock_dependencies['feature_computer'].validate_feature_definitions.return_value = {
            "rsi": True
        }

        sample_feature_request.skip_existing_features = False

        # Mock resampling returns no data
        mock_resampling_response = Mock()
        mock_resampling_response.resampled_data = {}  # Empty data
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = mock_resampling_response

        # When
        response = preprocess_service.process_feature_computation_request(sample_feature_request)

        # Then
        assert response.success is False
        assert "Market data resampling failed" in response.error_message

    def test_get_performance_summary(self, preprocess_service):
        """Test performance summary retrieval."""
        # Given
        # Simulate some processed requests
        preprocess_service._total_requests_processed = 5
        preprocess_service._total_features_computed = 50
        preprocess_service._total_processing_time_ms = 10000

        # When
        summary = preprocess_service.get_performance_summary()

        # Then
        assert summary['total_requests_processed'] == 5
        assert summary['total_features_computed'] == 50
        assert summary['average_processing_time_ms'] == 2000.0  # 10000 / 5
        assert summary['features_per_request_avg'] == 10.0  # 50 / 5

    def test_filter_features_to_compute_with_missing_features(
        self, preprocess_service, sample_feature_request
    ):
        """Test filtering when some features are missing."""
        # Given
        existing_features_info = {
            Timeframe.MINUTE_5: FeatureExistenceCheckResult(
                symbol="BTCUSDT",
                timeframe=Timeframe.MINUTE_5,
                existing_features=["sma"],  # Different feature exists
                missing_features=["rsi"],   # Our requested feature is missing
            )
        }

        # When
        filtered_features = preprocess_service._filter_features_to_compute(
            sample_feature_request, existing_features_info
        )

        # Then
        assert len(filtered_features) == 1  # Should compute all features
        assert filtered_features[0].name == "rsi"

    def test_filter_features_to_compute_all_exist(
        self, preprocess_service, sample_feature_request
    ):
        """Test filtering when all features already exist."""
        # Given
        existing_features_info = {
            Timeframe.MINUTE_5: FeatureExistenceCheckResult(
                symbol="BTCUSDT",
                timeframe=Timeframe.MINUTE_5,
                existing_features=["rsi"],  # Our requested feature exists
                missing_features=[],
            ),
            Timeframe.MINUTE_15: FeatureExistenceCheckResult(
                symbol="BTCUSDT",
                timeframe=Timeframe.MINUTE_15,
                existing_features=["rsi"],  # Our requested feature exists
                missing_features=[],
            )
        }

        # When
        filtered_features = preprocess_service._filter_features_to_compute(
            sample_feature_request, existing_features_info
        )

        # Then
        assert len(filtered_features) == 0  # No computation needed
