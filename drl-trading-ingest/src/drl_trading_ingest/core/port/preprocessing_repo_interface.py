"""
Port interface for preprocessing request repository operations.

This interface defines the contract for storing and retrieving
feature preprocessing requests.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest


class PreprocessingRepoPort(ABC):
    """
    Interface for preprocessing request repository operations.

    Defines the contract for storing and retrieving preprocessing requests
    with proper abstraction for testability.
    """

    @abstractmethod
    def save_request(self, request: FeaturePreprocessingRequest) -> None:
        """
        Store a preprocessing request.

        Args:
            request: The preprocessing request to store

        Raises:
            ValueError: If the request is invalid
            DatabaseConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    def get_request(self, request_id: str) -> Optional[FeaturePreprocessingRequest]:
        """
        Retrieve a preprocessing request by ID.

        Args:
            request_id: The unique identifier of the request

        Returns:
            FeaturePreprocessingRequest or None if not found
        """
        pass

    @abstractmethod
    def get_pending_requests(self, limit: int = 100) -> List[FeaturePreprocessingRequest]:
        """
        Get pending preprocessing requests.

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of pending preprocessing requests
        """
        pass

    @abstractmethod
    def update_request_status(self, request_id: str, status: str) -> None:
        """
        Update the status of a preprocessing request.

        Args:
            request_id: The unique identifier of the request
            status: New status for the request

        Raises:
            ValueError: If request not found or invalid status
        """
        pass
