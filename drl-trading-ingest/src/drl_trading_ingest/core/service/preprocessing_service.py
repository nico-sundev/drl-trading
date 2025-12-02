"""
Preprocessing service for handling feature preprocessing requests.

This service contains the business logic for processing feature preprocessing
requests, coordinating with various ports and adapters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from uuid import uuid4

from injector import inject

from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_ingest.core.port.preprocessing_repo_interface import PreprocessingRepoPort

logger = logging.getLogger("drl-trading-ingest")


class PreprocessingServiceInterface(ABC):
    """
    Interface for the preprocessing service.
    """

    @abstractmethod
    def submit_preprocessing_request(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit a feature preprocessing request for processing.

        Args:
            request_data: Dictionary containing the preprocessing request data

        Returns:
            Response dictionary with request_id and status
        """
        pass


@inject
class PreprocessingService(PreprocessingServiceInterface):
    """
    Service for handling feature preprocessing requests.

    This service validates requests, stores them for processing,
    and coordinates with other services for feature computation.
    """

    def __init__(self, preprocessing_repo: PreprocessingRepoPort):
        """
        Initialize the preprocessing service.

        Args:
            preprocessing_repo: Repository for storing preprocessing requests
        """
        self.preprocessing_repo = preprocessing_repo

    def submit_preprocessing_request(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit a feature preprocessing request for processing.

        This method validates the request, generates a request ID if needed,
        stores the request for later processing, and returns a response.

        Args:
            request_data: Dictionary containing the preprocessing request data

        Returns:
            Response dictionary with request_id and status

        Raises:
            ValueError: If the request data is invalid
        """
        try:
            # Validate and parse the request
            request = FeaturePreprocessingRequest(**request_data)

            # Generate request ID if not provided
            if not request.request_id:
                request.request_id = str(uuid4())

            logger.info("Received preprocessing request", extra={
                'request_id': request.request_id,
                'symbol': request.symbol,
                'base_timeframe': request.base_timeframe.value,
                'target_timeframes': [tf.value for tf in request.target_timeframes],
                'start_time': request.start_time.isoformat(),
                'end_time': request.end_time.isoformat(),
                'processing_context': request.processing_context
            })

            # Store the request for later processing
            self.preprocessing_repo.save_request(request)

            # TODO: Trigger async preprocessing workflow
            # For now, just acknowledge receipt

            logger.info("Preprocessing request accepted", extra={
                'request_id': request.request_id,
                'status': 'accepted'
            })

            return {
                "request_id": request.request_id,
                "status": "accepted"
            }

        except Exception as e:
            logger.error("Failed to process preprocessing request", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise ValueError(f"Invalid preprocessing request: {str(e)}")
