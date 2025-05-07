import logging
import os
from typing import Dict, List, Optional, Tuple

from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.base_agent import BaseAgent
from ai_trading.config.logging_config import configure_logging
from ai_trading.di.containers import ApplicationContainer

logger = logging.getLogger(__name__)


def bootstrap(
    config_path: Optional[str] = None,
) -> Tuple[ApplicationContainer, List[DataFrame]]:
    """Bootstrap the application using dependency injection.

    This function initializes the application container, configures logging,
    and processes data through the pipeline.

    Args:
        config_path: Optional path to config file. If not provided, will use default path.

    Returns:
        Tuple containing:
        - Application container with configured dependencies
        - List of preprocessed dataframes
    """
    # Initialize logging
    configure_logging()
    logger.info("Starting application bootstrap with dependency injection")

    # Create and configure the DI container
    container = ApplicationContainer()

    # Configure the container with the provided config path if specified
    if config_path:
        container.config_path.override(config_path)

    logger.info("DI container configured")

    # Initialize the feature config factory and discover config classes
    feature_config_factory = container.feature_config_factory()
    feature_config_factory.discover_config_classes()
    logger.info("Feature configuration classes discovered")

    # Parse feature parameters with the initialized factory
    features_config = container.features_config()
    features_config.parse_all_parameters(feature_config_factory)
    logger.info("Feature parameters parsed successfully")

    # Resolve services from container
    data_import_manager = container.data_import_manager()
    strip_service = container.strip_service()
    preprocess_service = container.preprocess_service()

    # Import data for all symbols
    symbol_containers = data_import_manager.get_data()
    logger.info(f"Imported data for {len(symbol_containers)} symbols")

    final_datasets = []
    for symbol_container in symbol_containers:
        logger.info(
            f"Processing {len(symbol_container.datasets)} datasets for symbol: {symbol_container.symbol}"
        )

        # Transform and strip other timeframes
        symbol_container.datasets = strip_service.strip_asset_price_datasets(
            symbol_container.datasets
        )
        logger.info("Timeframe stripping completed")

        # Preprocess data using injected service
        preprocessed_dataset = preprocess_service.preprocess_data(symbol_container)
        final_datasets.append(preprocessed_dataset)
        logger.info(
            f"Feature preprocessing completed for symbol: {symbol_container.symbol}"
        )

    logger.info("Application bootstrap completed successfully")
    return container, final_datasets


def create_environments_and_train(
    container: ApplicationContainer, final_datasets: List[DataFrame]
) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
    """
    Create environments and train agents using injected services.

    Args:
        container: Application container with resolved services
        final_datasets: List of preprocessed datasets to use for training

    Returns:
        Tuple containing:
        - Training environment (DummyVecEnv)
        - Validation environment (DummyVecEnv)
        - Dictionary of trained agents mapped by name
    """
    # Resolve required services
    split_service = container.split_service()
    agent_training_service = container.agent_training_service()

    # Split datasets
    split_datasets = []
    for dataset in final_datasets:
        split_datasets.append(split_service.split_dataset(dataset))

    # Create environments and train agents
    logger.info(
        f"Creating environments and training agents with {len(split_datasets)} datasets"
    )
    return agent_training_service.create_env_and_train_agents(split_datasets)


# Application entry point
if __name__ == "__main__":
    # Path is optional - default will be used from container configuration
    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs/applicationConfig.json"
    )

    # Bootstrap application with DI
    container, final_datasets = bootstrap(config_path)

    # Create environments and train agents
    train_env, val_env, agents = create_environments_and_train(
        container, final_datasets
    )
