"""
Ingest service bootstrap implementation.

Provides standardized service startup with hexagonal architecture compliance.
"""

import logging
import os
import time
from typing import Optional, List

from flask import Flask
from injector import Module

from drl_trading_common.infrastructure.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_common.infrastructure.health.health_check_service import HealthCheckService
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

logger = logging.getLogger(__name__)


class IngestServiceBootstrap(ServiceBootstrap):
    """
    Bootstrap implementation for the ingest service.

    Maintains hexagonal architecture boundaries with proper separation of:
    - Core business logic
    - Adapter layer (Flask routes, REST endpoints)
    - Infrastructure concerns (config, DI, logging)
    """

    def __init__(self) -> None:
        """Initialize the ingest service bootstrap."""
        # Set up environment for config loading before calling super().__init__
        # Use directory path so EnhancedServiceConfigLoader can auto-discover based on STAGE
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate from src/drl_trading_ingest/infrastructure/bootstrap/ to config directory
        # New hexagonal architecture: src/drl_trading_ingest/infrastructure/config/
        config_dir = os.path.join(os.path.dirname(current_dir), "config")

        os.environ["CONFIG_DIR"] = config_dir
        logger.info(f"Configuration directory set to: {config_dir}")

        # Use T004-compliant IngestConfig
        super().__init__(service_name="ingest", config_class=IngestConfig)
        self._app: Optional[Flask] = None
        self._health_service: Optional[HealthCheckService] = None

    def get_dependency_modules(self) -> List[Module]:
        """
        Return dependency injection modules for this service.

        Uses the existing IngestModule which properly wires the hexagonal architecture.
        """
        try:
            from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule
            return [IngestModule()]
        except ImportError as e:
            logger.error(f"Failed to import IngestModule: {e}")
            return []

    def _start_service(self) -> None:
        """
        Start service-specific logic.

        Creates the Flask application using the existing FlaskAppFactory.
        """
        try:
            # Use the existing FlaskAppFactory to create the app
            from drl_trading_ingest.infrastructure.bootstrap.flask_app_factory import FlaskAppFactory

            app = FlaskAppFactory.create_app(self.injector)
            self._app = app

            # Add health endpoint if not already present
            if not any(rule.rule == '/health' for rule in app.url_map.iter_rules()):
                @app.route('/health')
                def health():
                    return {"status": "healthy", "service": "drl-trading-ingest"}

            logger.info("Ingest service started successfully with FlaskAppFactory")

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise

    def _stop_service(self) -> None:
        """Stop service-specific logic."""
        logger.info("Stopping ingest service")
        try:
            if self._app:
                logger.info("Flask application stopped")
        except Exception as e:
            logger.error(f"Error stopping service: {e}")

    def _run_main_loop(self) -> None:
        """
        Run the main service loop.

        For Flask services, this starts the web server.
        """
        if self._app:
            # In production, use a proper WSGI server like gunicorn
            # For development, run directly
            port = int(os.environ.get("PORT", 8080))
            debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

            logger.info(f"Starting Flask application on port {port}")
            self._app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)
        else:
            logger.error("No Flask application to run")
            # Keep the service alive if no app
            while self.is_running:
                time.sleep(1)

    def get_app(self) -> Optional[Flask]:
        """Get the Flask application instance."""
        return self._app
