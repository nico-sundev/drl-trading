"""
Route registrar for DRL Trading Ingest service.

Handles registration of service-specific routes for data ingestion endpoints.
"""

import logging
from flask import Flask
from injector import Injector

from drl_trading_common.application.web.generic_flask_app_factory import RouteRegistrar

logger = logging.getLogger(__name__)


class IngestRouteRegistrar(RouteRegistrar):
    """Route registrar for ingest service specific endpoints."""

    def register_routes(self, app: Flask, injector: Injector) -> bool:
        """
        Register ingest service specific routes.

        Args:
            app: Flask application instance
            injector: Dependency injection container

        Returns:
            bool: True if routes were registered successfully, False otherwise
        """
        try:
            # Import and register existing ingest routes
            from drl_trading_ingest.adapter.web.routes import register_routes
            register_routes(app, injector)
            logger.info("Ingest service routes registered successfully")
            return True
        except ImportError as e:
            logger.warning(f"Could not import ingest routes: {e}")
            return False
        except Exception as e:
            # Log a concise error message to avoid redundancy with underlying service errors
            logger.error(f"Failed to register ingest routes: {type(e).__name__}")
            # Don't re-raise - let the Flask factory handle it gracefully
            # The service can continue running without routes if needed
            return False
