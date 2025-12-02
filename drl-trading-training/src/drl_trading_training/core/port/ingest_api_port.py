"""
Port interfaces for external services.

This module defines the interfaces (ports) that the training service uses
to communicate with external systems, following hexagonal architecture principles.
"""

from abc import ABC, abstractmethod
from typing import Dict

from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest


class IngestApiPort(ABC):
    """
    Port for communicating with the ingest service API.

    This interface defines the contract for submitting feature preprocessing requests
    to the ingest service via REST API.
    """

    @abstractmethod
    def submit_preprocessing_request(self, request_data: FeaturePreprocessingRequest) -> Dict[str, str]:
        """
        Submit a feature preprocessing request to the ingest service.

        Args:
            request_data: The preprocessing request data as a dictionary

        Returns:
            Response from the ingest service containing request_id and status
        """
        pass
