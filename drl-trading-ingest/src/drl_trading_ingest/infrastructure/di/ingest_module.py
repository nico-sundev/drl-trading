import logging
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from flask import Flask
from injector import Binder, Injector, Module, provider, singleton
from confluent_kafka import Producer

from drl_trading_ingest.adapter.migration.alembic_migration_service import (
    AlembicMigrationService,
)
from drl_trading_ingest.adapter.rest.ingestion_controller import (
    IngestionController,
    IngestionControllerInterface,
)
from drl_trading_ingest.adapter.timescale.market_data_repo import MarketDataRepo
from drl_trading_ingest.core.port.market_data_repo_interface import (
    MarketDataRepoPort,
)
from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationServiceInterface,
)
from drl_trading_ingest.core.service.ingestion_service import (
    IngestionService,
    IngestionServiceInterface,
)
from drl_trading_ingest.infrastructure.bootstrap.flask_app_factory import (
    FlaskAppFactory,
)
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

logger = logging.getLogger(__name__)


class IngestModule(Module):
    """Dependency injection module for ingest service using provided config.

    The bootstrap now supplies the authoritative config instance; we avoid
    re-loading configuration here to ensure consistency and eliminate IO.
    """

    def __init__(self, config: IngestConfig) -> None:
        self._config = config

    @provider
    def provide_ingest_config(self) -> IngestConfig:
        """Provide the already loaded IngestConfig instance."""
        return self._config

    @provider
    @singleton
    def provide_kafka_producer(
        self, application_config: IngestConfig
    ) -> Producer:
        """Provide a Kafka producer instance."""
        # KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
        producer = Producer({
            'bootstrap.servers': application_config.infrastructure.messaging.host
        })
        return producer

    @provider
    @singleton
    def provide_flask_app(self, injector: Injector) -> Flask:
        """
        Provide a Flask application instance using the factory pattern.

        This is now properly separated - the factory handles infrastructure
        concerns while routes handle adapter concerns.
        """
        return FlaskAppFactory.create_app(injector)

    @provider
    @singleton
    def provide_session_factory(self, config: IngestConfig) -> SQLAlchemySessionFactory:
        """Provide a SQLAlchemy session factory instance."""
        return SQLAlchemySessionFactory(config.infrastructure.database)

    def configure(self, binder: Binder) -> None:
        """Configure the module with necessary bindings."""
        # Core services
        binder.bind(IngestionServiceInterface, to=IngestionService, scope=singleton)

        # Adapters
        binder.bind(IngestionControllerInterface, to=IngestionController, scope=singleton)
        binder.bind(MarketDataRepoPort, to=MarketDataRepo, scope=singleton)
        binder.bind(MigrationServiceInterface, to=AlembicMigrationService, scope=singleton)
