"""
Kafka implementation of preprocessing message publisher.

This adapter uses KafkaProducerAdapter to publish preprocessing completion and error
notifications to Kafka topics with retry logic and resilience behavior.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from drl_trading_common.adapter.messaging.kafka_producer_adapter import (
    KafkaProducerAdapter,
)
from drl_trading_common.model.feature_preprocessing_request import (
    FeaturePreprocessingRequest,
)
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
    PreprocessingMessagePublisherPort,
)


class KafkaPreprocessingMessagePublisher(PreprocessingMessagePublisherPort):
    """
    Kafka-based message publisher for preprocessing notifications.

    This implementation publishes preprocessing completion and error events
    to Kafka topics using the KafkaProducerAdapter for retry logic and resilience.

    Attributes:
        _completion_producer: Producer for preprocessing completion events.
        _error_producer: Producer for error events (DLQ).
        _completion_topic: Topic name for completion events.
        _error_topic: Topic name for errors (DLQ).
    """

    def __init__(
        self,
        completion_producer: KafkaProducerAdapter,
        error_producer: KafkaProducerAdapter,
        completion_topic: str,
        error_topic: str,
    ) -> None:
        """
        Initialize Kafka preprocessing message publisher.

        Args:
            completion_producer: Producer for completion events (with appropriate retry config).
            error_producer: Producer for errors (typically minimal retry for DLQ).
            completion_topic: Topic name for publishing completion events.
            error_topic: Topic name for publishing errors (DLQ).
        """
        self.logger = logging.getLogger(__name__)
        self._completion_producer = completion_producer
        self._error_producer = error_producer
        self._completion_topic = completion_topic
        self._error_topic = error_topic

        self.logger.info(
            f"KafkaPreprocessingMessagePublisher initialized: "
            f"completion_topic={completion_topic}, "
            f"error_topic={error_topic}"
        )

    def publish_preprocessing_completed(
        self,
        request: FeaturePreprocessingRequest,
        processing_context: str,
        total_features_computed: int,
        timeframes_processed: List[Timeframe],
        success_details: Dict[str, Any],
    ) -> None:
        """
        Publish preprocessing completion event to Kafka.

        Serializes the completion details to JSON and publishes to the configured
        topic. Uses request_id as the message key for traceability.

        Args:
            request: Original preprocessing request.
            processing_context: Processing mode ("training", "inference", "backfill").
            total_features_computed: Number of features computed across all timeframes.
            timeframes_processed: List of timeframes that were successfully processed.
            success_details: Additional details about the successful processing.

        Raises:
            KafkaException: If publishing fails after all retries.
        """
        payload = {
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": processing_context,
            "total_features_computed": total_features_computed,
            "timeframes_processed": [tf.value for tf in timeframes_processed],
            "feature_definitions_count": len(request.get_enabled_features()),
            "target_timeframes": [tf.value for tf in request.target_timeframes],
            "base_timeframe": request.base_timeframe.value,
            "start_time": request.start_time.isoformat(),
            "end_time": request.end_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "success_details": success_details,
        }

        self._completion_producer.publish(
            topic=self._completion_topic,
            key=request.request_id,  # Partition by request_id for traceability
            value=payload,
            headers={
                "event_type": "preprocessing_completed",
                "symbol": request.symbol,
                "processing_context": processing_context,
                "request_id": request.request_id,
            },
        )

        self.logger.info(
            f"Published preprocessing completion: {request.symbol} "
            f"(Request: {request.request_id}, Context: {processing_context}) "
            f"- {total_features_computed} features across {len(timeframes_processed)} timeframes"
        )

    def publish_preprocessing_error(
        self,
        request: FeaturePreprocessingRequest,
        processing_context: str,
        error_message: str,
        error_details: Dict[str, str],
        failed_step: str,
    ) -> None:
        """
        Publish preprocessing error to Kafka DLQ.

        Serializes error information to JSON and publishes to the error topic.
        Uses request_id as the message key for consistency.

        Args:
            request: Original preprocessing request.
            processing_context: Processing mode ("training", "inference", "backfill").
            error_message: Human-readable error message.
            error_details: Detailed error information for debugging.
            failed_step: Which step of the preprocessing pipeline failed.

        Raises:
            KafkaException: If publishing to DLQ fails after all retries.
        """
        payload = {
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": processing_context,
            "error_message": error_message,
            "error_details": error_details,
            "failed_step": failed_step,
            "target_timeframes": [tf.value for tf in request.target_timeframes],
            "base_timeframe": request.base_timeframe.value,
            "start_time": request.start_time.isoformat(),
            "end_time": request.end_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }

        self._error_producer.publish(
            topic=self._error_topic,
            key=request.request_id,
            value=payload,
            headers={
                "event_type": "preprocessing_error",
                "symbol": request.symbol,
                "processing_context": processing_context,
                "request_id": request.request_id,
                "failed_step": failed_step,
            },
        )

        self.logger.error(
            f"Published preprocessing error for {request.symbol} "
            f"(Request: {request.request_id}, Context: {processing_context}) "
            f"at step '{failed_step}': {error_message}"
        )

    def publish_feature_validation_error(
        self,
        request: FeaturePreprocessingRequest,
        invalid_features: List[str],
        validation_errors: Dict[str, str],
    ) -> None:
        """
        Publish feature validation error to Kafka DLQ.

        Serializes validation error information to JSON and publishes to the error topic.
        Uses request_id as the message key for consistency.

        Args:
            request: Original preprocessing request.
            invalid_features: List of feature names that failed validation.
            validation_errors: Mapping of feature names to validation error messages.

        Raises:
            KafkaException: If publishing to DLQ fails after all retries.
        """
        payload = {
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": request.processing_context,
            "invalid_features": invalid_features,
            "validation_errors": validation_errors,
            "total_features_requested": len(request.get_enabled_features()),
            "timestamp": datetime.now().isoformat(),
        }

        self._error_producer.publish(
            topic=self._error_topic,
            key=request.request_id,
            value=payload,
            headers={
                "event_type": "feature_validation_error",
                "symbol": request.symbol,
                "processing_context": request.processing_context,
                "request_id": request.request_id,
            },
        )

        self.logger.error(
            f"Published feature validation error for {request.symbol} "
            f"(Request: {request.request_id}) "
            f"- {len(invalid_features)} invalid features: {invalid_features}"
        )

    def health_check(self) -> bool:
        """
        Check if Kafka producers are healthy.

        Returns:
            True if both producers are healthy, False otherwise.
        """
        completion_healthy = self._completion_producer.health_check()
        error_healthy = self._error_producer.health_check()

        is_healthy = completion_healthy and error_healthy

        if not is_healthy:
            self.logger.warning(
                f"Health check failed: "
                f"completion_producer={completion_healthy}, "
                f"error_producer={error_healthy}"
            )

        return is_healthy

    def close(self) -> None:
        """Close both Kafka producers and flush pending messages."""
        self.logger.info("Closing KafkaPreprocessingMessagePublisher...")
        self._completion_producer.close()
        self._error_producer.close()
        self.logger.info("KafkaPreprocessingMessagePublisher closed successfully")
