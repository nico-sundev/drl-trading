"""
In-memory repository implementation for preprocessing requests.

This is a temporary implementation for development and testing.
In production, this would be replaced with a proper database-backed repository.
"""

import logging
from typing import Dict, List, Optional

from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_ingest.core.port.preprocessing_repo_interface import PreprocessingRepoPort

logger = logging.getLogger("drl-trading-ingest")


class InMemoryPreprocessingRepo(PreprocessingRepoPort):
    """
    In-memory implementation of preprocessing repository.

    Stores requests in memory for development and testing purposes.
    Not suitable for production use.
    """

    def __init__(self):
        """Initialize the in-memory repository."""
        self._requests: Dict[str, FeaturePreprocessingRequest] = {}
        logger.info("Initialized in-memory preprocessing repository")

    def save_request(self, request: FeaturePreprocessingRequest) -> None:
        """
        Store a preprocessing request in memory.

        Args:
            request: The preprocessing request to store

        Raises:
            ValueError: If the request is invalid
        """
        if not request.request_id:
            raise ValueError("Request must have a valid request_id")

        self._requests[request.request_id] = request
        logger.debug("Saved preprocessing request", extra={
            'request_id': request.request_id,
            'symbol': request.symbol
        })

    def get_request(self, request_id: str) -> Optional[FeaturePreprocessingRequest]:
        """
        Retrieve a preprocessing request by ID.

        Args:
            request_id: The unique identifier of the request

        Returns:
            FeaturePreprocessingRequest or None if not found
        """
        return self._requests.get(request_id)

    def get_pending_requests(self, limit: int = 100) -> List[FeaturePreprocessingRequest]:
        """
        Get pending preprocessing requests.

        For now, returns all stored requests (no status tracking yet).

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of preprocessing requests
        """
        requests = list(self._requests.values())
        return requests[:limit]

    def update_request_status(self, request_id: str, status: str) -> None:
        """
        Update the status of a preprocessing request.

        Note: Current implementation doesn't track status, just logs the update.

        Args:
            request_id: The unique identifier of the request
            status: New status for the request

        Raises:
            ValueError: If request not found
        """
        if request_id not in self._requests:
            raise ValueError(f"Request {request_id} not found")

        logger.info("Updated preprocessing request status", extra={
            'request_id': request_id,
            'new_status': status
        })
