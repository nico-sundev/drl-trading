"""
Unit tests for KafkaPreprocessingMessagePublisher.

Tests cover initialization, publishing completion events, publishing errors,
health checks, and resource cleanup.
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from builders import FeaturePreprocessingRequestBuilder
from drl_trading_common.adapter.model.feature_preprocessing_request import (
    FeaturePreprocessingRequest,
)
from drl_trading_common.adapter.model.timeframe import Timeframe
from drl_trading_common.core.model.processing_context import ProcessingContext
from drl_trading_preprocess.adapter.messaging.publisher.kafka_preprocessing_message_publisher import (
    KafkaPreprocessingMessagePublisher,
)


# Test fixtures
@pytest.fixture
def mock_completion_producer() -> Mock:
    """Create mock KafkaProducerAdapter for completion events."""
    producer = Mock()
    producer.publish = Mock()
    producer.health_check = Mock(return_value=True)
    producer.close = Mock()
    return producer


@pytest.fixture
def mock_error_producer() -> Mock:
    """Create mock KafkaProducerAdapter for error events."""
    producer = Mock()
    producer.publish = Mock()
    producer.health_check = Mock(return_value=True)
    producer.close = Mock()
    return producer


@pytest.fixture
def completion_topic() -> str:
    """Kafka topic for completion events."""
    return "preprocessing.completed"


@pytest.fixture
def error_topic() -> str:
    """Kafka topic for error events (DLQ)."""
    return "preprocessing.errors"


@pytest.fixture
def sample_request() -> FeaturePreprocessingRequest:
    """Create sample preprocessing request for testing."""
    return (
        FeaturePreprocessingRequestBuilder()
        .with_request_id("test-request-123")
        .with_symbol("EURUSD")
        .with_base_timeframe(Timeframe.MINUTE_1)
        .with_target_timeframes([Timeframe.MINUTE_5, Timeframe.MINUTE_15])
        .with_time_range(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 23, 59, 59),
        )
        .with_processing_context(ProcessingContext.TRAINING.value)
        .with_feature_names("sma_20", "rsi_14")
        .build()
    )


class TestKafkaPreprocessingMessagePublisherInitialization:
    """Test publisher initialization."""

    def test_initialization_with_valid_producers(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test initialization with valid Kafka producers."""
        # Given / When
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        # Then
        assert publisher._completion_producer == mock_completion_producer
        assert publisher._error_producer == mock_error_producer
        assert publisher._completion_topic == completion_topic
        assert publisher._error_topic == error_topic


class TestKafkaPreprocessingMessagePublisherPublishCompleted:
    """Test publishing preprocessing completion events."""

    def test_publish_preprocessing_completed_success(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
        sample_request: FeaturePreprocessingRequest,
    ) -> None:
        """Test successful publishing of preprocessing completion event."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        timeframes_processed = [Timeframe.MINUTE_5, Timeframe.MINUTE_15]
        success_details: Dict[str, Any] = {
            "compute_duration_ms": 1234,
            "storage_duration_ms": 567,
        }

        # When
        publisher.publish_preprocessing_completed(
            request=sample_request,
            processing_context=ProcessingContext.TRAINING.value,
            total_features_computed=100,
            timeframes_processed=timeframes_processed,
            success_details=success_details,
        )

        # Then
        mock_completion_producer.publish.assert_called_once()
        call_args = mock_completion_producer.publish.call_args

        # Verify topic and key
        assert call_args.kwargs["topic"] == completion_topic
        assert call_args.kwargs["key"] == sample_request.request_id

        # Verify payload
        payload = call_args.kwargs["value"]
        assert payload["request_id"] == sample_request.request_id
        assert payload["symbol"] == sample_request.symbol
        assert payload["processing_context"] == "training"
        assert payload["total_features_computed"] == 100
        assert payload["timeframes_processed"] == ["5m", "15m"]
        assert payload["feature_definitions_count"] == 2
        assert payload["target_timeframes"] == ["5m", "15m"]
        assert payload["base_timeframe"] == "1m"
        assert "timestamp" in payload
        assert payload["success_details"] == success_details

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert headers["event_type"] == "preprocessing_completed"
        assert headers["symbol"] == sample_request.symbol
        assert headers["processing_context"] == "training"
        assert headers["request_id"] == sample_request.request_id

    def test_publish_preprocessing_completed_with_multiple_timeframes(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
        sample_request: FeaturePreprocessingRequest,
    ) -> None:
        """Test completion event with multiple timeframes."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        timeframes_processed = [
            Timeframe.MINUTE_5,
            Timeframe.MINUTE_15,
            Timeframe.MINUTE_30,
        ]

        # When
        publisher.publish_preprocessing_completed(
            request=sample_request,
            processing_context=ProcessingContext.INFERENCE.value,
            total_features_computed=250,
            timeframes_processed=timeframes_processed,
            success_details={},
        )

        # Then
        payload = mock_completion_producer.publish.call_args.kwargs["value"]
        assert len(payload["timeframes_processed"]) == 3
        assert payload["timeframes_processed"] == ["5m", "15m", "30m"]
        assert payload["total_features_computed"] == 250


class TestKafkaPreprocessingMessagePublisherPublishError:
    """Test publishing preprocessing error events."""

    def test_publish_preprocessing_error_success(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
        sample_request: FeaturePreprocessingRequest,
    ) -> None:
        """Test successful publishing of preprocessing error event."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        error_details = {
            "error_type": "ComputationError",
            "traceback": "some traceback here",
        }

        # When
        publisher.publish_preprocessing_error(
            request=sample_request,
            processing_context=ProcessingContext.TRAINING.value,
            error_message="Failed to compute features",
            error_details=error_details,
            failed_step="feature_computation",
        )

        # Then
        mock_error_producer.publish.assert_called_once()
        call_args = mock_error_producer.publish.call_args

        # Verify topic and key
        assert call_args.kwargs["topic"] == error_topic
        assert call_args.kwargs["key"] == sample_request.request_id

        # Verify payload
        payload = call_args.kwargs["value"]
        assert payload["request_id"] == sample_request.request_id
        assert payload["symbol"] == sample_request.symbol
        assert payload["processing_context"] == "training"
        assert payload["error_message"] == "Failed to compute features"
        assert payload["error_details"] == error_details
        assert payload["failed_step"] == "feature_computation"
        assert payload["target_timeframes"] == ["5m", "15m"]
        assert payload["base_timeframe"] == "1m"
        assert "timestamp" in payload

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert headers["event_type"] == "preprocessing_error"
        assert headers["symbol"] == sample_request.symbol
        assert headers["failed_step"] == "feature_computation"


class TestKafkaPreprocessingMessagePublisherPublishValidationError:
    """Test publishing feature validation error events."""

    def test_publish_feature_validation_error_success(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
        sample_request: FeaturePreprocessingRequest,
    ) -> None:
        """Test successful publishing of feature validation error."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        invalid_features = ["invalid_feature_1", "invalid_feature_2"]
        validation_errors = {
            "invalid_feature_1": "Feature not supported",
            "invalid_feature_2": "Invalid parameters",
        }

        # When
        publisher.publish_feature_validation_error(
            request=sample_request,
            invalid_features=invalid_features,
            validation_errors=validation_errors,
        )

        # Then
        mock_error_producer.publish.assert_called_once()
        call_args = mock_error_producer.publish.call_args

        # Verify topic and key
        assert call_args.kwargs["topic"] == error_topic
        assert call_args.kwargs["key"] == sample_request.request_id

        # Verify payload
        payload = call_args.kwargs["value"]
        assert payload["request_id"] == sample_request.request_id
        assert payload["symbol"] == sample_request.symbol
        assert payload["processing_context"] == sample_request.processing_context
        assert payload["invalid_features"] == invalid_features
        assert payload["validation_errors"] == validation_errors
        assert payload["total_features_requested"] == 2
        assert "timestamp" in payload

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert headers["event_type"] == "feature_validation_error"
        assert headers["symbol"] == sample_request.symbol


class TestKafkaPreprocessingMessagePublisherHealthCheck:
    """Test publisher health checks."""

    def test_health_check_both_producers_healthy(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test health check when both producers are healthy."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        mock_completion_producer.health_check.return_value = True
        mock_error_producer.health_check.return_value = True

        # When
        is_healthy = publisher.health_check()

        # Then
        assert is_healthy is True
        mock_completion_producer.health_check.assert_called_once()
        mock_error_producer.health_check.assert_called_once()

    def test_health_check_completion_producer_unhealthy(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test health check when completion producer is unhealthy."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        mock_completion_producer.health_check.return_value = False
        mock_error_producer.health_check.return_value = True

        # When
        is_healthy = publisher.health_check()

        # Then
        assert is_healthy is False

    def test_health_check_error_producer_unhealthy(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test health check when error producer is unhealthy."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        mock_completion_producer.health_check.return_value = True
        mock_error_producer.health_check.return_value = False

        # When
        is_healthy = publisher.health_check()

        # Then
        assert is_healthy is False

    def test_health_check_both_producers_unhealthy(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test health check when both producers are unhealthy."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        mock_completion_producer.health_check.return_value = False
        mock_error_producer.health_check.return_value = False

        # When
        is_healthy = publisher.health_check()

        # Then
        assert is_healthy is False


class TestKafkaPreprocessingMessagePublisherClose:
    """Test publisher resource cleanup."""

    def test_close_calls_both_producers(
        self,
        mock_completion_producer: Mock,
        mock_error_producer: Mock,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """Test close calls close on both producers."""
        # Given
        publisher = KafkaPreprocessingMessagePublisher(
            completion_producer=mock_completion_producer,
            error_producer=mock_error_producer,
            completion_topic=completion_topic,
            error_topic=error_topic,
        )

        # When
        publisher.close()

        # Then
        mock_completion_producer.close.assert_called_once()
        mock_error_producer.close.assert_called_once()
