"""
Port for publishing preprocessing messages to messaging infrastructure.

This port defines the interface for publishing preprocessing completion and error
notifications to external systems (typically Kafka). It supports the fire-and-forget
architecture by providing async notifications about preprocessing results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.computation.feature_preprocessing_request import FeaturePreprocessingRequest


class PreprocessingMessagePublisherPort(ABC):
    """
    Abstract port for publishing preprocessing messages.

    This port enables the preprocessing service to publish notifications about:
    - Successful feature computation completion
    - Processing errors and failures
    - Feature validation errors

    Implementations can target different messaging systems (Kafka, RabbitMQ, etc.)
    or provide stubs for development/testing.
    """

    @abstractmethod
    def publish_preprocessing_completed(
        self,
        request: FeaturePreprocessingRequest,
        processing_duration_seconds: float,
        total_features_computed: int,
        timeframes_processed: List[Timeframe],
        success_details: Dict[str, Any]
    ) -> None:
        """
        Publish notification about successful preprocessing completion.

        Args:
            request: Original preprocessing request
            processing_duration_seconds: Total processing time in seconds
            total_features_computed: Number of features computed across all timeframes
            timeframes_processed: List of timeframes that were successfully processed
            success_details: Additional details about the successful processing
        """
        pass

    @abstractmethod
    def publish_preprocessing_error(
        self,
        request: FeaturePreprocessingRequest,
        processing_duration_seconds: float,
        error_message: str,
        error_details: Dict[str, str],
        failed_step: str
    ) -> None:
        """
        Publish notification about preprocessing failure.

        Args:
            request: Original preprocessing request
            processing_duration_seconds: Processing time until failure
            error_message: Human-readable error message
            error_details: Detailed error information for debugging
            failed_step: Which step of the preprocessing pipeline failed
        """
        pass

    @abstractmethod
    def publish_feature_validation_error(
        self,
        request: FeaturePreprocessingRequest,
        invalid_features: List[str],
        validation_errors: Dict[str, str]
    ) -> None:
        """
        Publish notification about feature validation failure.

        Args:
            request: Original preprocessing request
            invalid_features: List of feature names that failed validation
            validation_errors: Mapping of feature names to validation error messages
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the message publisher is healthy and can publish messages.

        Returns:
            True if healthy, False if there are connectivity or other issues
        """
        pass
