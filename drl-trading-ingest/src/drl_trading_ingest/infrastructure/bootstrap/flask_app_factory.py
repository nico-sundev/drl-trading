"""
Flask application factory for the DRL Trading Ingest service.

This module provides infrastructure-level Flask application creation
and configuration, separate from route definitions which are adapters.
"""

import logging
from typing import Optional

from flask import Flask, request
from injector import Injector

from drl_trading_ingest.adapter.web.routes import register_routes
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

logger = logging.getLogger(__name__)


class FlaskAppFactory:
    """
    Factory for creating and configuring Flask applications.

    This is infrastructure-level code that handles application setup,
    middleware configuration, and error handling.
    """

    @staticmethod
    def create_app(injector: Injector, config: Optional[IngestConfig] = None) -> Flask:
        """
        Create and configure a Flask application instance.

        Args:
            injector: Dependency injection container
            config: Optional configuration override

        Returns:
            Flask: Configured Flask application
        """
        app = Flask(__name__)

        if config is None:
            config = injector.get(IngestConfig)

        # Configure Flask from our config
        FlaskAppFactory._configure_flask_settings(app, config)

        # Set up error handling
        FlaskAppFactory._configure_error_handling(app)

        # Set up logging
        FlaskAppFactory._configure_logging(app)

        # Register routes (primary adapters)
        register_routes(app, injector)

        logger.info("Flask application created and configured successfully")
        return app

    @staticmethod
    def _configure_flask_settings(app: Flask, config: IngestConfig) -> None:
        """Configure Flask-specific settings."""
        app.config.update({
            'DEBUG': False,  # Never debug in production
            'TESTING': False,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': False,
        })

    @staticmethod
    def _configure_error_handling(app: Flask) -> None:
        """Configure application-wide error handling."""
        @app.errorhandler(404)
        def not_found(error):
            return {"error": "Endpoint not found", "service": "drl-trading-ingest"}, 404

        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return {"error": "Internal server error", "service": "drl-trading-ingest"}, 500

    @staticmethod
    def _configure_logging(app: Flask) -> None:
        """Configure request logging."""
        @app.before_request
        def log_request_info() -> None:
            logger.debug(f"Request: {request.method} {request.url}")

        @app.after_request
        def log_response_info(response):
            logger.debug(f"Response: {response.status_code}")
            return response
