import logging
import os

from drl_trading_common.config.service_config_loader import ServiceConfigLoader
from flask import Flask
from injector import Module, provider, singleton
from kafka import KafkaProducer

from drl_trading_ingest.adapter.rest.ingestion_controller import (
    IngestionController,
    IngestionControllerInterface,
)
from drl_trading_ingest.adapter.timescale.timescale_repo import (
    TimescaleRepo,
    TimescaleRepoInterface,
)
from drl_trading_ingest.core.ingestion_service import (
    IngestionService,
    IngestionServiceInterface,
)
from drl_trading_ingest.infrastructure.config.data_ingestion_config import (
    DataIngestionConfig,
)

logger = logging.getLogger(__name__)


class IngestModule(Module):

    @provider
    def provide_data_ingestion_config(self) -> DataIngestionConfig:
        """Provide the DataIngestionConfig instance."""
        config_path = os.environ.get("SERVICE_CONFIG_PATH")
        if not config_path:
            raise ValueError("SERVICE_CONFIG_PATH environment variable is not set.")

        logger.info(f"Loading configuration from SERVICE_CONFIG_PATH: {config_path}")
        config = ServiceConfigLoader.load_config(
            DataIngestionConfig, config_path=config_path
        )
        return config

    @provider
    @singleton
    def provide_kafka_producer(
        self, application_config: DataIngestionConfig
    ) -> KafkaProducer:
        """Provide a Kafka producer instance."""
        # KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")
        producer = KafkaProducer(
            bootstrap_servers=[application_config.infrastructure.messaging.host]
        )
        return producer

    @provider
    @singleton
    def provide_flask_app(self) -> Flask:
        """Provide a Flask application instance."""
        from flask import Flask

        app = Flask(__name__)
        return app

    def configure(self, binder) -> None:
        """Configure the module with necessary bindings."""
        binder.bind(
            IngestionControllerInterface, to=IngestionController, scope=singleton
        )
        binder.bind(TimescaleRepoInterface, to=TimescaleRepo, scope=singleton)
        binder.bind(IngestionServiceInterface, to=IngestionService, scope=singleton)
