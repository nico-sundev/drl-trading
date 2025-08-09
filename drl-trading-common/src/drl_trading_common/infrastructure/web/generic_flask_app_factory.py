"""
Generic Flask application factory for DRL Trading services.

This module provides infrastructure-level Flask application creation
and configuration that can be reused across all microservices.
"""

import logging
from typing import Optional
from abc import ABC, abstractmethod

from flask import Flask, request, jsonify
from injector import Injector

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.infrastructure.health.health_check_service import (
    HealthCheckService,
)

logger = logging.getLogger(__name__)


class RouteRegistrar(ABC):
    """Abstract base class for service-specific route registration."""

    @abstractmethod
    def register_routes(self, app: Flask, injector: Injector) -> bool:
        """
        Register service-specific routes.

        Args:
            app: Flask application instance
            injector: Dependency injection container

        Returns:
            bool: True if routes were registered successfully, False otherwise
        """
        pass


class GenericFlaskAppFactory:
    """
    Generic factory for creating and configuring Flask applications.

    This is infrastructure-level code that handles application setup,
    middleware configuration, and error handling for any DRL Trading service.
    """

    @staticmethod
    def create_app(
        service_name: str,
        injector: Injector,
        config: Optional[BaseApplicationConfig] = None,
        route_registrar: Optional[RouteRegistrar] = None,
    ) -> Flask:
        """
        Create and configure a Flask application instance.

        Args:
            service_name: Name of the service (e.g., "ingest", "training")
            injector: Dependency injection container
            config: Optional configuration override
            route_registrar: Optional service-specific route registrar

        Returns:
            Flask: Configured Flask application
        """
        app = Flask(f"drl-trading-{service_name}")

        if config is None:
            try:
                config = injector.get(BaseApplicationConfig)
            except Exception as e:
                logger.warning(f"Could not get config from injector: {e}")
                config = None

        # Configure Flask from our config
        GenericFlaskAppFactory._configure_flask_settings(app, config)

        # Set up error handling
        GenericFlaskAppFactory._configure_error_handling(app, service_name)

        # Set up logging
        GenericFlaskAppFactory._configure_logging(app)

        # Register health endpoints (standard for all services)
        GenericFlaskAppFactory._register_health_endpoints(app, injector, service_name)

        # Register service-specific routes if provided
        if route_registrar:
            try:
                success = route_registrar.register_routes(app, injector)
                if success:
                    logger.info(f"Service-specific routes registered for {service_name}")
                else:
                    logger.warning(f"Some routes could not be registered for {service_name} (service may have limited functionality)")
            except Exception as e:
                logger.warning(
                    f"Could not register service routes for {service_name}: {str(type(e).__name__)} (see service logs for details)"
                )

        logger.info(
            f"Flask application created and configured successfully for {service_name}"
        )
        return app

    @staticmethod
    def _configure_flask_settings(
        app: Flask, config: Optional[BaseApplicationConfig]
    ) -> None:
        """Configure Flask-specific settings."""
        app.config.update(
            {
                "DEBUG": False,  # Never debug in production
                "TESTING": False,
                "JSON_SORT_KEYS": False,
                "JSONIFY_PRETTYPRINT_REGULAR": False,
            }
        )

    @staticmethod
    def _configure_error_handling(app: Flask, service_name: str) -> None:
        """Configure application-wide error handling."""

        @app.errorhandler(404)
        def not_found(error):
            return (
                jsonify(
                    {
                        "error": "Endpoint not found",
                        "service": f"drl-trading-{service_name}",
                    }
                ),
                404,
            )

        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return (
                jsonify(
                    {
                        "error": "Internal server error",
                        "service": f"drl-trading-{service_name}",
                    }
                ),
                500,
            )

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

    @staticmethod
    def _register_health_endpoints(
        app: Flask, injector: Injector, service_name: str
    ) -> None:
        """Register standardized health check endpoints."""

        @app.route("/health")
        def health():
            """Basic health endpoint."""
            try:
                health_service = injector.get(HealthCheckService)
                result = health_service.check_health()
                status_code = 200 if result["status"] == "healthy" else 503
                return jsonify(result), status_code
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return (
                    jsonify(
                        {
                            "status": "unhealthy",
                            "message": f"Health check failed: {str(e)}",
                            "service": f"drl-trading-{service_name}",
                        }
                    ),
                    503,
                )

        @app.route("/health/ready")
        def readiness():
            """Kubernetes readiness probe endpoint."""
            try:
                health_service = injector.get(HealthCheckService)
                result = health_service.check_readiness()
                status_code = 200 if result.get("ready", False) else 503
                return jsonify(result), status_code
            except Exception as e:
                logger.error(f"Readiness check failed: {e}")
                return (
                    jsonify(
                        {
                            "ready": False,
                            "status": "unhealthy",
                            "message": f"Readiness check failed: {str(e)}",
                            "service": f"drl-trading-{service_name}",
                        }
                    ),
                    503,
                )

        @app.route("/health/live")
        def liveness():
            """Kubernetes liveness probe endpoint."""
            return (
                jsonify({"status": "alive", "service": f"drl-trading-{service_name}"}),
                200,
            )


class DefaultRouteRegistrar(RouteRegistrar):
    """Default route registrar that registers no additional routes."""

    def register_routes(self, app: Flask, injector: Injector) -> None:
        """Register no additional routes by default."""
        pass
