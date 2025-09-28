"""
Tests for PreprocessService fire-and-forget architecture.

Tests the refactored service that no longer returns responses but operates
in a fire-and-forget pattern with async Kafka notifications.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.computation.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.core.service.preprocess_service import PreprocessService


class TestPreprocessServiceFireAndForget:
    """Test the fire-and-forget preprocess service behavior."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for PreprocessService."""
        return {
            'market_data_resampler': Mock(),
            'feature_computer': Mock(),
            'feature_validator': Mock(),
            'feature_store_port': Mock(),
            'feature_existence_checker': Mock(),
            'message_publisher': Mock(),
        }

    @pytest.fixture
    def preprocess_service(self, mock_dependencies):
        """Create PreprocessService with mocked dependencies."""
        return PreprocessService(
            market_data_resampler=mock_dependencies['market_data_resampler'],
            feature_computer=mock_dependencies['feature_computer'],
            feature_validator=mock_dependencies['feature_validator'],
            feature_store_port=mock_dependencies['feature_store_port'],
            feature_existence_checker=mock_dependencies['feature_existence_checker'],
            message_publisher=mock_dependencies['message_publisher'],
        )

    @pytest.fixture
    def sample_request(self):
        """Create a sample feature preprocessing request."""
        feature_def = FeatureDefinition(
            name="test_feature",
            enabled=True,
            derivatives=[0],
            parameter_sets=[]
        )

        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[feature_def.dict()],
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5],
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
            description="Test features"
        )

        return FeaturePreprocessingRequest(
            symbol="BTCUSD",
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5],
            feature_definitions=[feature_def],
            feature_config_version_info=feature_version_info,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
            request_id="test_request_123",
        )

    def test_process_feature_computation_request_returns_none(self, preprocess_service, sample_request):
        """Test that the main processing method returns None (fire-and-forget)."""
        # Given
        # Mock all the internal method calls to avoid complex setup
        preprocess_service._validate_request = Mock()
        preprocess_service._check_existing_features = Mock(return_value={})
        preprocess_service._filter_features_to_compute = Mock(return_value=sample_request.get_enabled_features())
        preprocess_service._handle_feature_warmup = Mock(return_value=True)
        preprocess_service._resample_market_data = Mock(return_value={})
        preprocess_service._compute_features_for_timeframes = Mock(return_value={})
        preprocess_service._store_computed_features = Mock()

        # When
        result = preprocess_service.process_feature_computation_request(sample_request)

        # Then
        assert result is None, "Fire-and-forget service should return None"

    def test_early_return_when_no_features_to_compute(self, preprocess_service, sample_request):
        """Test early return when no features need computation."""
        # Given
        preprocess_service._validate_request = Mock()
        preprocess_service._check_existing_features = Mock(return_value={})
        preprocess_service._filter_features_to_compute = Mock(return_value=[])  # No features to compute
        preprocess_service._resample_market_data = Mock()  # Mock this method to track calls

        # When
        result = preprocess_service.process_feature_computation_request(sample_request)

        # Then
        assert result is None, "Should return None when no features to compute"
        # Should not call downstream methods
        preprocess_service._resample_market_data.assert_not_called()

    def test_early_return_when_warmup_fails(self, preprocess_service, sample_request):
        """Test early return when feature warmup fails."""
        # Given
        preprocess_service._validate_request = Mock()
        preprocess_service._check_existing_features = Mock(return_value={})
        preprocess_service._filter_features_to_compute = Mock(return_value=sample_request.get_enabled_features())
        preprocess_service._handle_feature_warmup = Mock(return_value=False)  # Warmup fails

        # When
        result = preprocess_service.process_feature_computation_request(sample_request)

        # Then
        assert result is None, "Should return None when warmup fails"

    @patch('drl_trading_preprocess.core.service.preprocess_service.logger')
    def test_error_handling_logs_and_returns_none(self, mock_logger, preprocess_service, sample_request):
        """Test that exceptions are logged but method still returns None."""
        # Given
        preprocess_service._validate_request = Mock(side_effect=Exception("Test error"))

        # When
        result = preprocess_service.process_feature_computation_request(sample_request)

        # Then
        assert result is None, "Should return None even when exception occurs"
        mock_logger.error.assert_called_once()

    def test_performance_metrics_updated_on_success(self, preprocess_service, sample_request):
        """Test that performance metrics are updated on successful completion."""
        # Given
        initial_requests = preprocess_service._total_requests_processed
        initial_features = preprocess_service._total_features_computed

        preprocess_service._validate_request = Mock()
        preprocess_service._check_existing_features = Mock(return_value={})
        preprocess_service._filter_features_to_compute = Mock(return_value=sample_request.get_enabled_features())
        preprocess_service._handle_feature_warmup = Mock(return_value=True)
        preprocess_service._resample_market_data = Mock(return_value={Timeframe.MINUTE_5: []})
        preprocess_service._compute_features_for_timeframes = Mock(return_value={Timeframe.MINUTE_5: Mock(columns=['feat1', 'feat2'])})
        preprocess_service._store_computed_features = Mock()

        # When
        preprocess_service.process_feature_computation_request(sample_request)

        # Then
        assert preprocess_service._total_requests_processed == initial_requests + 1
        assert preprocess_service._total_features_computed > initial_features

    def test_get_performance_summary_works(self, preprocess_service):
        """Test that performance summary can be retrieved."""
        # When
        summary = preprocess_service.get_performance_summary()

        # Then
        assert isinstance(summary, dict)
        assert 'total_requests_processed' in summary
        assert 'total_features_computed' in summary
        assert 'average_processing_time_ms' in summary
        assert 'features_per_request_avg' in summary
