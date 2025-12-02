"""
REST API adapter for communicating with the ingest service.

This adapter implements the IngestApiPort interface using the generated
OpenAPI client for the ingest service API.
"""

from typing import Dict

from injector import inject

from drl_trading_training.core.port.ingest_api_port import IngestApiPort

from .generated.ingest_api_client import ApiClient, DefaultApi
from .generated.ingest_api_client.models import FeaturePreprocessingRequest as GeneratedFeaturePreprocessingRequest

from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest


class IngestApiAdapter(IngestApiPort):
    """
    Adapter for the ingest service REST API.

    Uses the generated OpenAPI client to communicate with the ingest service.
    """

    @inject
    def __init__(self, api_client: ApiClient = None):
        """
        Initialize the adapter with an API client.

        Args:
            api_client: Configured API client for the ingest service.
                       If None, creates a default client.
        """
        if api_client is None:
            # TODO: Configure with proper base URL from environment/config
            api_client = ApiClient()
            api_client.configuration.host = "http://localhost:5000/api/v1"

        self.api = DefaultApi(api_client)

    def submit_preprocessing_request(self, request_data: FeaturePreprocessingRequest) -> Dict[str, str]:
        """
        Submit a feature preprocessing request to the ingest service.

        Args:
            request_data: The preprocessing request data

        Returns:
            Response containing request_id and status
        """
        # Convert Pydantic model to generated model
        request = GeneratedFeaturePreprocessingRequest(**request_data.model_dump(mode="json"))

        # Call the API
        response = self.api.submit_preprocessing_request(request)

        # Convert response back to dict
        return {
            "request_id": response.request_id,
            "status": response.status
        }
