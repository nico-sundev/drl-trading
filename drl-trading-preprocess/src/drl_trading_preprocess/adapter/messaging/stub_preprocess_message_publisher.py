"""
Stub implementation of preprocessing message publisher for development and testing.

This adapter provides a non-functional stub implementation of the PreprocessingMessagePublisherPort
that logs messages instead of actually publishing them to Kafka. This enables development
and testing of the preprocessing service without requiring Kafka infrastructure.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.computation.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import PreprocessingMessagePublisherPort


class StubPreprocessingMessagePublisher(PreprocessingMessagePublisherPort):
    """
    Stub message publisher for preprocessing service development and testing.

    This implementation logs all publishing operations instead of actually
    sending messages to Kafka infrastructure. Useful for:
    - Local development without Kafka
    - Unit testing preprocessing logic
    - Integration testing feature computation pipeline
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize stub publisher with configurable logging.

        Args:
            log_level: Logging level for published messages (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self._published_messages: List[Dict] = []
        self._error_messages: List[Dict] = []
        self._is_healthy = True

    def publish_preprocessing_completed(
        self,
        request: FeaturePreprocessingRequest,
        processing_duration_seconds: float,
        total_features_computed: int,
        timeframes_processed: List[Timeframe],
        success_details: Dict[str, Any]
    ) -> None:
        """
        Log preprocessing completion (stub implementation).

        Stores message details for inspection and logs summary information.
        """
        message = {
            "type": "preprocessing_completed",
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": request.processing_context,
            "processing_duration_seconds": processing_duration_seconds,
            "total_features_computed": total_features_computed,
            "timeframes_processed": [tf.value for tf in timeframes_processed],
            "feature_definitions_count": len(request.get_enabled_features()),
            "target_timeframes": [tf.value for tf in request.target_timeframes],
            "base_timeframe": request.base_timeframe.value,
            "start_time": request.start_time.isoformat(),
            "end_time": request.end_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "success_details": success_details
        }

        self._published_messages.append(message)

        self.logger.info(
            f"[STUB] Preprocessing completed: {request.symbol} "
            f"(Request: {request.request_id}) "
            f"- {total_features_computed} features across {len(timeframes_processed)} timeframes "
            f"in {processing_duration_seconds:.2f}s"
        )

        # Log detailed breakdown per timeframe
        for timeframe in timeframes_processed:
            features_for_tf = success_details.get(f"features_{timeframe.value}", "unknown")
            self.logger.debug(
                f"[STUB] {request.symbol}:{timeframe.value} - "
                f"{features_for_tf} features computed"
            )

    def publish_preprocessing_error(
        self,
        request: FeaturePreprocessingRequest,
        processing_duration_seconds: float,
        error_message: str,
        error_details: Dict[str, str],
        failed_step: str
    ) -> None:
        """
        Log preprocessing error (stub implementation).

        Stores error details for inspection and logs error information.
        """
        error_msg = {
            "type": "preprocessing_error",
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": request.processing_context,
            "processing_duration_seconds": processing_duration_seconds,
            "error_message": error_message,
            "error_details": error_details,
            "failed_step": failed_step,
            "target_timeframes": [tf.value for tf in request.target_timeframes],
            "base_timeframe": request.base_timeframe.value,
            "start_time": request.start_time.isoformat(),
            "end_time": request.end_time.isoformat(),
            "timestamp": datetime.now().isoformat()
        }

        self._error_messages.append(error_msg)

        self.logger.error(
            f"[STUB] Preprocessing failed: {request.symbol} "
            f"(Request: {request.request_id}) "
            f"at step '{failed_step}' after {processing_duration_seconds:.2f}s: {error_message}"
        )

        if error_details:
            self.logger.error(f"[STUB] Error details: {error_details}")

    def publish_feature_validation_error(
        self,
        request: FeaturePreprocessingRequest,
        invalid_features: List[str],
        validation_errors: Dict[str, str]
    ) -> None:
        """
        Log feature validation error (stub implementation).

        Stores validation error details for inspection and logs error information.
        """
        error_msg = {
            "type": "feature_validation_error",
            "request_id": request.request_id,
            "symbol": request.symbol,
            "processing_context": request.processing_context,
            "invalid_features": invalid_features,
            "validation_errors": validation_errors,
            "total_features_requested": len(request.get_enabled_features()),
            "timestamp": datetime.now().isoformat()
        }

        self._error_messages.append(error_msg)

        self.logger.error(
            f"[STUB] Feature validation failed: {request.symbol} "
            f"(Request: {request.request_id}) "
            f"- {len(invalid_features)} invalid features: {invalid_features}"
        )

        for feature, error in validation_errors.items():
            self.logger.error(f"[STUB] Validation error for '{feature}': {error}")

    def health_check(self) -> bool:
        """
        Return stub health status.

        Always returns True unless explicitly set to unhealthy for testing.
        """
        self.logger.debug("[STUB] Health check - preprocessing message publisher stub is always healthy")
        return self._is_healthy

    # Additional methods for testing and inspection

    def get_published_messages(self) -> List[Dict]:
        """Get all published messages for testing inspection."""
        return self._published_messages.copy()

    def get_error_messages(self) -> List[Dict]:
        """Get all error messages for testing inspection."""
        return self._error_messages.copy()

    def get_completion_messages(self) -> List[Dict]:
        """Get all completion messages for testing inspection."""
        return [msg for msg in self._published_messages if msg["type"] == "preprocessing_completed"]

    def get_validation_error_messages(self) -> List[Dict]:
        """Get all validation error messages for testing inspection."""
        return [msg for msg in self._error_messages if msg["type"] == "feature_validation_error"]

    def clear_messages(self) -> None:
        """Clear all stored messages (useful for test cleanup)."""
        self._published_messages.clear()
        self._error_messages.clear()

    def set_health_status(self, is_healthy: bool) -> None:
        """Set health status for testing failure scenarios."""
        self._is_healthy = is_healthy
        self.logger.info(f"[STUB] Preprocessing message publisher health status set to: {is_healthy}")

    def get_message_count(self) -> Dict[str, int]:
        """Get count of different message types."""
        completion_count = len([msg for msg in self._published_messages if msg["type"] == "preprocessing_completed"])
        error_count = len([msg for msg in self._error_messages if msg["type"] == "preprocessing_error"])
        validation_error_count = len([msg for msg in self._error_messages if msg["type"] == "feature_validation_error"])

        return {
            "preprocessing_completed": completion_count,
            "preprocessing_errors": error_count,
            "validation_errors": validation_error_count,
            "total": len(self._published_messages) + len(self._error_messages)
        }

    def get_symbols_processed(self) -> List[str]:
        """Get list of symbols that have been processed."""
        symbols = set()
        for msg in self._published_messages:
            symbols.add(msg["symbol"])
        for msg in self._error_messages:
            symbols.add(msg["symbol"])
        return sorted(list(symbols))

    def get_latest_completion_for_symbol(self, symbol: str) -> Dict[str, Any] | None:
        """Get the latest completion message for a specific symbol."""
        completion_messages = [
            msg for msg in self._published_messages
            if msg["type"] == "preprocessing_completed" and msg["symbol"] == symbol
        ]

        if not completion_messages:
            return None

        return max(completion_messages, key=lambda x: x["timestamp"])

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics across all messages."""
        completion_messages = self.get_completion_messages()

        if not completion_messages:
            return {
                "total_requests": 0,
                "avg_processing_time": 0.0,
                "total_features_computed": 0,
                "success_rate": 0.0
            }

        total_requests = len(completion_messages) + len(self._error_messages)
        total_duration = sum(msg["processing_duration_seconds"] for msg in completion_messages)
        total_features = sum(msg["total_features_computed"] for msg in completion_messages)
        success_rate = len(completion_messages) / total_requests * 100 if total_requests > 0 else 0.0

        return {
            "total_requests": total_requests,
            "successful_requests": len(completion_messages),
            "avg_processing_time": total_duration / len(completion_messages) if completion_messages else 0.0,
            "total_features_computed": total_features,
            "success_rate": success_rate
        }
