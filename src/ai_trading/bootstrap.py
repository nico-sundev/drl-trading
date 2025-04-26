import logging
import os
from typing import Dict, Tuple

from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.base_agent import BaseAgent
from ai_trading.config.application_config import ApplicationConfig
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.logging_config import configure_logging
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.data_set_utils.split_service import SplitService
from ai_trading.data_set_utils.strip_service import StripService
from ai_trading.model.split_dataset_container import SplitDataSetContainer
from ai_trading.preprocess.feast.feast_service import FeastService
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.preprocess_service import PreprocessService
from ai_trading.services.agent_training_service import AgentTrainingService

logger = logging.getLogger(__name__)


def bootstrap(config: ApplicationConfig) -> list[DataFrame]:
    """Bootstrap the application."""
    # Initialize logging
    configure_logging()
    logger.info("Starting application bootstrap")

    # Load configuration
    logger.info("Configuration loaded successfully")

    # Create a single CsvDataImportService with the complete config
    data_import_svc = CsvDataImportService(config.local_data_import_config)

    # Import data for all symbols
    symbol_containers = data_import_svc.import_data()

    strip_svc = StripService()
    logger.debug("Strip service initialized")

    feature_class_registry = FeatureClassRegistry()
    logger.debug("Feature class registry initialized")

    feast_svc = FeastService(feature_store_config=config.feature_store_config)
    logger.debug("Feast service initialized")

    feature_aggregator = FeatureAggregator(
        config=config.features_config,
        class_registry=feature_class_registry,
        feast_service=feast_svc,
    )
    logger.debug("Feature aggregator initialized")

    # Preprocess the asset price datasets with feature store support
    preprocess_svc = PreprocessService(
        features_config=config.features_config,
        feature_class_registry=feature_class_registry,
        feature_aggregator=feature_aggregator,
    )

    final_datasets = []
    for symbol_container in symbol_containers:
        logger.info(
            f"Imported {len(symbol_container.datasets)} datasets for symbol: {symbol_container.symbol}"
        )
        for dataset in symbol_container.datasets:
            logger.info(f"Imported dataset: {dataset.timeframe}")

        # Transform and strip other timeframes using the stripper service
        symbol_container.datasets = strip_svc.strip_asset_price_datasets(
            symbol_container.datasets
        )
        logger.info("Timeframe stripping completed")

        preprocessed_dataset = preprocess_svc.preprocess_data(symbol_container)
        final_datasets.append(preprocessed_dataset)
        logger.info(
            f"Feature preprocessing completed for symbol: {symbol_container.symbol}"
        )

    return final_datasets


def _split_datasets(
    final_datasets: list[DataFrame], config: ApplicationConfig
) -> list[SplitDataSetContainer]:
    split_svc = SplitService(config.rl_model_config)
    return [
        *(split_svc.split_dataset(df=final_dataset) for final_dataset in final_datasets)
    ]


def create_environments_and_train(
    final_datasets: list[DataFrame], config: ApplicationConfig
) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
    """Create environments and train agents.

    Args:
        base_dataset: Preprocessed dataset to use for training and validation
        config: Application configuration containing environment and model settings

    Returns:
        Tuple containing:
        - Training environment (DummyVecEnv)
        - Validation environment (DummyVecEnv)
        - Dictionary of trained agents mapped by name
    """
    # Create the environment and train the agents using the factory pattern
    training_svc = AgentTrainingService(config=config)

    return training_svc.create_env_and_train_agents(
        _split_datasets(final_datasets, config)
    )


# Application entry point
if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs/applicationConfig.json"
    )
    config = ConfigLoader.get_config(config_path)
    final_datasets = bootstrap(config)

    train_env, val_env, agents = create_environments_and_train(final_datasets, config)
