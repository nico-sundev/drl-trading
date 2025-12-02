"""
REST controller for feature preprocessing requests.

This controller handles HTTP requests for submitting feature preprocessing
requests, following the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod

from flask import jsonify, request
from flask.views import MethodView
from injector import inject

from drl_trading_ingest.core.service.preprocessing_service import PreprocessingServiceInterface


class PreprocessingControllerInterface(ABC):
    """
    Interface for the preprocessing controller.
    """

    @abstractmethod
    def submit_preprocessing_request(self):
        """
        Handle submission of feature preprocessing request.
        """
        ...


@inject
class PreprocessingController(MethodView, PreprocessingControllerInterface):
    """
    Controller for handling feature preprocessing requests.

    This controller receives HTTP requests for feature preprocessing,
    validates them, and delegates to the preprocessing service.
    """

    def __init__(self, preprocessing_service: PreprocessingServiceInterface):
        """
        Initialize the controller with dependencies.

        Args:
            preprocessing_service: Service for handling preprocessing logic
        """
        self.preprocessing_service = preprocessing_service

    def post(self):
        """
        Handle POST request for submitting preprocessing request.

        Expects JSON payload with FeaturePreprocessingRequest structure.
        Returns JSON response with request_id and status.
        """
        try:
            # Get JSON data from request
            request_data = request.get_json()

            if not request_data:
                return jsonify({
                    "error": "Request body is required",
                    "service": "drl-trading-ingest"
                }), 400

            # Delegate to service
            result = self.preprocessing_service.submit_preprocessing_request(request_data)

            # Return success response
            return jsonify(result), 202  # 202 Accepted

        except ValueError as e:
            # Validation error
            return jsonify({
                "error": f"Invalid request format: {str(e)}",
                "service": "drl-trading-ingest"
            }), 400

        except Exception:
            # Unexpected error
            return jsonify({
                "error": "Internal server error",
                "service": "drl-trading-ingest"
            }), 500
