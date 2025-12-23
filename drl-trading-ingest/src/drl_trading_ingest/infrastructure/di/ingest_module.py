import logging

from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from flask import Flask
from injector import Binder, Injector, Module, provider, singleton

from drl_trading_ingest.adapter.data_import.local import CsvDataImportService
from drl_trading_ingest.adapter.data_import.web import (
    BinanceDataProvider,
    TwelveDataProvider,
    YahooDataImportService,
)
from drl_trading_ingest.adapter.migration.alembic_migration_service import (
    AlembicMigrationService,
)
from drl_trading_ingest.adapter.memory.preprocessing_repo import (
    InMemoryPreprocessingRepo,
)
from drl_trading_ingest.adapter.rest.preprocessing_controller import (
    PreprocessingController,
    PreprocessingControllerInterface,
)
from drl_trading_ingest.adapter.timescale.market_data_repo import MarketDataRepo
from drl_trading_ingest.core.port.market_data_repo_interface import (
    MarketDataRepoPort,
)
from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationServiceInterface,
)
from drl_trading_ingest.core.port.preprocessing_repo_interface import (
    PreprocessingRepoPort,
)
from drl_trading_ingest.core.service.preprocessing_service import (
    PreprocessingService,
    PreprocessingServiceInterface,
)
from drl_trading_ingest.core.service.data_provider_manager import DataProviderManager
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

    # Provider registry mapping: config_key -> (provider_class, config_accessor)
    _PROVIDER_REGISTRY = [
        ('csv', CsvDataImportService, lambda cfg: cfg.data_source.csv),
        ('binance', BinanceDataProvider, lambda cfg: cfg.data_source.binance),
        ('twelve_data', TwelveDataProvider, lambda cfg: cfg.data_source.twelve_data),
        ('yahoo', YahooDataImportService, lambda cfg: cfg.data_source.yahoo),
    ]

    @provider
    @singleton
    def provide_data_provider_manager(self, config: IngestConfig) -> DataProviderManager:
        """Initialize and return data provider manager with registered providers.

        Automatically discovers and registers all providers defined in _PROVIDER_REGISTRY.
        """
        manager = DataProviderManager()

        logger.info("Initializing data providers from configuration...")

        for key, provider_class, config_accessor in self._PROVIDER_REGISTRY:
            provider_config = config_accessor(config)

            if not provider_config.enabled:
                continue

            try:
                provider_instance = provider_class(provider_config.model_dump())
                provider_instance.setup()
                manager.register_provider(key, provider_instance)
                logger.info(f"Initialized provider: {key}")
            except Exception as e:
                logger.error(f"Failed to initialize {key} provider: {e}")

        logger.info(f"Initialized {len(manager.get_available_provider_names())} data providers")
        return manager

    def configure(self, binder: Binder) -> None:
        """Configure the module with necessary bindings."""
        # Core services
        binder.bind(PreprocessingServiceInterface, to=PreprocessingService, scope=singleton)

        # Adapters
        binder.bind(PreprocessingControllerInterface, to=PreprocessingController, scope=singleton)
        binder.bind(MarketDataRepoPort, to=MarketDataRepo, scope=singleton)
        binder.bind(PreprocessingRepoPort, to=InMemoryPreprocessingRepo, scope=singleton)
        binder.bind(MigrationServiceInterface, to=AlembicMigrationService, scope=singleton)
