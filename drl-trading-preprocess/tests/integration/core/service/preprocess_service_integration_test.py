"""
Integration tests for PreprocessService with stub message publisher.

Tests the complete preprocessing pipeline including async message publishing
using the stub implementation for development verification.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.adapter.messaging.stub_preprocess_message_publisher import StubPreprocessingMessagePublisher
from drl_trading_preprocess.core.model.computation.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.core.service.preprocess_service import PreprocessService


class TestPreprocessServiceWithStubPublisher:
    """Test preprocessing service with stub message publisher integration."""

    @pytest.fixture
    def stub_message_publisher(self):
        """Create stub message publisher for testing."""
        return StubPreprocessingMessagePublisher(log_level="DEBUG")

    @pytest.fixture
    def mock_dependencies(self, stub_message_publisher):
        """Create mock dependencies with real stub publisher."""
        return {
            'market_data_resampler': Mock(),
            'feature_computer': Mock(),
            'feature_validator': Mock(),
            'feature_store_port': Mock(),
            'feature_existence_checker': Mock(),
            'message_publisher': stub_message_publisher,
        }

    @pytest.fixture
    def preprocess_service(self, mock_dependencies):
        """Create PreprocessService with stub message publisher."""
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

    def test_successful_processing_publishes_completion_message(
        self,
        preprocess_service,
        sample_request,
        stub_message_publisher
    ):
        """Test that successful processing publishes completion message."""
        # Given
        preprocess_service._validate_request = Mock()
        preprocess_service._check_existing_features = Mock(return_value={})
        preprocess_service._filter_features_to_compute = Mock(return_value=sample_request.get_enabled_features())
        preprocess_service._handle_feature_warmup = Mock(return_value=True)
        preprocess_service._resample_market_data = Mock(return_value={Timeframe.MINUTE_5: Mock()})
        mock_dataframe = Mock()
        mock_dataframe.columns = ['feat1', 'feat2']  # Configure columns attribute
        mock_dataframe.__len__ = Mock(return_value=100)  # DataFrame has 100 rows
        preprocess_service._compute_features_for_timeframes = Mock(
            return_value={Timeframe.MINUTE_5: mock_dataframe}
        )
        preprocess_service._store_computed_features = Mock()

        # When
        preprocess_service.process_feature_computation_request(sample_request)

        # Then
        completion_messages = stub_message_publisher.get_completion_messages()
        assert len(completion_messages) == 1

        message = completion_messages[0]
        assert message["type"] == "preprocessing_completed"
        assert message["request_id"] == "test_request_123"
        assert message["symbol"] == "BTCUSD"
        assert message["total_features_computed"] == 2  # This counts feature columns
        assert len(message["timeframes_processed"]) == 1
        assert message["timeframes_processed"][0] == "5m"

    def test_processing_error_publishes_error_message(
        self,
        preprocess_service,
        sample_request,
        stub_message_publisher
    ):
        """Test that processing errors publish error messages."""
        # Given
        preprocess_service._validate_request = Mock(side_effect=Exception("Test processing error"))

        # When
        preprocess_service.process_feature_computation_request(sample_request)

        # Then
        error_messages = stub_message_publisher.get_error_messages()
        assert len(error_messages) == 1

        message = error_messages[0]
        assert message["type"] == "preprocessing_error"
        assert message["request_id"] == "test_request_123"
        assert message["symbol"] == "BTCUSD"
        assert message["error_message"] == "Test processing error"
        assert message["failed_step"] == "processing_pipeline"

    def test_validation_error_publishes_validation_error_message(
        self,
        preprocess_service,
        sample_request,
        stub_message_publisher
    ):
        """Test that validation errors publish validation error messages."""
        # Given
        preprocess_service.feature_validator.validate_definitions = Mock(
            return_value={"test_feature": False}
        )

        # When
        preprocess_service.process_feature_computation_request(sample_request)

        # Then
        validation_errors = stub_message_publisher.get_validation_error_messages()
        assert len(validation_errors) == 1

        message = validation_errors[0]
        assert message["type"] == "feature_validation_error"
        assert message["request_id"] == "test_request_123"
        assert message["symbol"] == "BTCUSD"
        assert message["invalid_features"] == ["test_feature"]

    def test_stub_publisher_provides_processing_stats(self, stub_message_publisher):
        """Test that stub publisher provides useful processing statistics."""
        # Given - no messages yet
        initial_stats = stub_message_publisher.get_processing_stats()

        # When - simulate some processing
        stub_message_publisher._published_messages.append({
            "type": "preprocessing_completed",
            "symbol": "BTCUSD",
            "processing_duration_seconds": 1.5,
            "total_features_computed": 5
        })
        stub_message_publisher._published_messages.append({
            "type": "preprocessing_completed",
            "symbol": "ETHUSD",
            "processing_duration_seconds": 2.0,
            "total_features_computed": 8
        })

        # Then
        assert initial_stats["total_requests"] == 0
        assert initial_stats["success_rate"] == 0.0

        final_stats = stub_message_publisher.get_processing_stats()
        assert final_stats["total_requests"] == 2
        assert final_stats["successful_requests"] == 2
        assert final_stats["avg_processing_time"] == 1.75
        assert final_stats["total_features_computed"] == 13
        assert final_stats["success_rate"] == 100.0

    def test_stub_publisher_tracks_symbols_processed(self, stub_message_publisher):
        """Test that stub publisher tracks which symbols have been processed."""
        # Given - simulate processing multiple symbols
        stub_message_publisher._published_messages.append({
            "type": "preprocessing_completed",
            "symbol": "BTCUSD"
        })
        stub_message_publisher._error_messages.append({
            "type": "preprocessing_error",
            "symbol": "ETHUSD"
        })

        # When
        symbols = stub_message_publisher.get_symbols_processed()

        # Then
        assert symbols == ["BTCUSD", "ETHUSD"]

    def test_stub_publisher_health_check(self, stub_message_publisher):
        """Test stub publisher health check functionality."""
        # Given
        assert stub_message_publisher.health_check() is True

        # When - simulate unhealthy state
        stub_message_publisher.set_health_status(False)

        # Then
        assert stub_message_publisher.health_check() is False
