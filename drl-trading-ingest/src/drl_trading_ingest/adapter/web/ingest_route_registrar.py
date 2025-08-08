"""
Route registrar for DRL Trading Ingest service.

Handles registration of service-specific routes for data ingestion endpoints.
"""

import logging
from flask import Flask
from injector import Injector

from drl_trading_common.infrastructure.web.generic_flask_app_factory import RouteRegistrar

logger = logging.getLogger(__name__)


class IngestRouteRegistrar(RouteRegistrar):
    """Route registrar for ingest service specific endpoints."""

    def register_routes(self, app: Flask, injector: Injector) -> None:
        """
        Register ingest service specific routes.

        Args:
            app: Flask application instance
            injector: Dependency injection container
        """
        try:
            # Import and register existing ingest routes
            from drl_trading_ingest.adapter.web.routes import register_routes
            register_routes(app, injector)
            logger.info("Ingest service routes registered successfully")
        except ImportError as e:
            logger.warning(f"Could not import ingest routes: {e}")
        except Exception as e:
            logger.error(f"Failed to register ingest routes: {e}")
            raise
