import logging
import os
from typing import Dict, List, Tuple

from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.base_agent import BaseAgent
from ai_trading.config.application_config import ApplicationConfig
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.config.logging_config import configure_logging
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.data_set_utils.split_service import SplitService
from ai_trading.data_set_utils.timeframe_stripper_service import (
    TimeframeStripperService,
)
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.split_dataset_container import SplitDataSetContainer
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.preprocess_service import PreprocessService
from ai_trading.services.agent_training_service import AgentTrainingService

logger = logging.getLogger(__name__)


def bootstrap(config: ApplicationConfig) -> DataFrame:
    """Bootstrap the application."""
    # Initialize logging
    configure_logging()
    logger.info("Starting application bootstrap")

    # Load configuration
    logger.info("Configuration loaded successfully")

    # Initialize services and aggregate datasets from all symbols
    all_datasets: List[AssetPriceDataSet] = []

    # Create a single CsvDataImportService with the complete config
    data_import_svc = CsvDataImportService(config.local_data_import_config)

    # Import data for all symbols
    symbol_containers = data_import_svc.import_data()

    # Extract datasets from all symbols
    for symbol_container in symbol_containers:
        all_datasets.extend(symbol_container.datasets)
        logger.info(f"Imported datasets for symbol {symbol_container.symbol}")

    logger.info(f"Imported {len(all_datasets)} raw asset price datasets in total")

    # Transform and strip other timeframes using the stripper service
    tf_stripper_svc = TimeframeStripperService()
    stripped_raw_asset_price_datasets = tf_stripper_svc.strip_asset_price_datasets(
        all_datasets
    )
    logger.info("Timeframe stripping completed")

    # Initialize the feature class registry
    feature_class_registry = FeatureClassRegistry()
    logger.info("Feature class registry initialized")

    # Preprocess the asset price datasets with feature store support
    preprocess_svc = PreprocessService(
        datasets=stripped_raw_asset_price_datasets,
        features_config=config.features_config,
        feature_class_registry=feature_class_registry,
        feature_store_config=config.feature_store_config,
    )
    preprocessed_dataset = preprocess_svc.preprocess_data()
    logger.info("Feature preprocessing completed")

    return preprocessed_dataset


def create_environments_and_train(
    base_dataset: DataFrame, config: ApplicationConfig
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
    # Split the preprocessed data into training, validation, and test sets
    data_set_prep_svc = SplitService()
    data_sets: SplitDataSetContainer = data_set_prep_svc.split_dataset(
        df=base_dataset, split_ratios=(0.8, 0.1, 0.1)  # train, val, test ratios
    )

    # Create the environment and train the agents using the factory pattern
    training_svc = AgentTrainingService(env_config=config.environment_config)

    return training_svc.create_env_and_train_agents(
        data_sets.training_data,
        data_sets.validation_data,
        config.rl_model_config.total_timesteps,
        config.rl_model_config.agent_threshold,
        config.rl_model_config.agents,
    )


# Application entry point
if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs/applicationConfig.json"
    )
    config = ConfigLoader.get_config(config_path)
    base_dataset = bootstrap(config)

    train_env, val_env, agents = create_environments_and_train(base_dataset, config)
