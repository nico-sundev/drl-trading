import logging

from flask import Flask
from injector import Injector

from drl_trading_ingest.adapter.web.routes import register_routes
from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationServiceInterface,
)
from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule

logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Flask application with dependency injection.

    This function sets up the complete application stack including:
    - Dependency injection container
    - Database migration management
    - Route registration

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    # Initialize dependency injection container
    injector = Injector([IngestModule()])

    # Ensure database migrations are applied on startup
    try:
        migration_service = injector.get(MigrationServiceInterface)
        migration_service.ensure_migrations_applied()
        logger.info("Database migrations successfully ensured")
    except Exception as e:
        logger.error(f"Failed to ensure migrations on startup: {str(e)}")
        # Consider whether to continue or fail hard based on your requirements
        # For production, you might want to fail hard:
        # raise

    # Register application routes
    register_routes(app, injector)

    return app


if __name__ == "__main__":
    # Configure logging for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = create_app()
    app.run(debug=True)
