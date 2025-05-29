"""Modern bootstrap using injector-based dependency injection.

This module replaces the old dependency-injector based bootstrap with
a Spring-like annotation-based dependency injection system.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Type

from drl_trading_common.base.base_trading_env import BaseTradingEnv
from drl_trading_common.config.logging_config import configure_logging

from drl_trading_framework.common.agents.base_agent import BaseAgent
from drl_trading_framework.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_framework.common.di.domain_module import get_trading_injector
from drl_trading_framework.common.model.preprocessing_result import PreprocessingResult
from drl_trading_framework.inference.inference_service import InferenceService
from drl_trading_framework.preprocess.data_set_utils.split_service import SplitService
from drl_trading_framework.preprocess.data_set_utils.strip_service import StripService
from drl_trading_framework.preprocess.preprocess_service import PreprocessService
from drl_trading_framework.training.services.agent_training_service import (
    AgentTrainingService,
)

logger = logging.getLogger(__name__)


# Updated _preprocess to use modern injector-based DI
def _preprocess(
    feature_class_discovery_package: Optional[str] = None,
    feature_config_discovery_package: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[None, List[PreprocessingResult]]:
    """Bootstrap the application using modern dependency injection.

    This function initializes the injector container, configures logging,
    and processes data through the pipeline using @inject decorators.

    Args:
        config_path: Optional path to config file. If provided, it overrides any
                     path set by DRL_TRADING_CONFIG_PATH environment variable or default.

    Returns:
        Tuple containing:
        - None (no longer returning container, services are resolved via injector)
        - List of preprocessed dataframes
    """
    # Initialize logging
    configure_logging()
    logger.info("Starting application bootstrap with modern dependency injection")

    # Get the trading injector with optional config path
    injector = get_trading_injector(config_path)

    # Log the effective config path being used
    env_var_value = os.getenv("DRL_TRADING_CONFIG_PATH")
    if config_path:
        logger.info(f"Using explicit config_path parameter: {config_path}")
    elif env_var_value:
        logger.info(
            f"Using config path from DRL_TRADING_CONFIG_PATH environment variable: {env_var_value}"
        )
    else:
        logger.info("Using default config path from injector configuration")

    logger.info("Modern DI injector configured")

    # Handle feature discovery if needed
    if feature_class_discovery_package:
        logger.info(
            f"Discovering feature classes from package: {feature_class_discovery_package}"
        )
        # TODO: Integrate feature discovery with injector if needed

    if feature_config_discovery_package:
        logger.info(
            f"Discovering feature config classes from package: {feature_config_discovery_package}"
        )
        # TODO: Integrate feature config discovery with injector if needed

    # Resolve services from injector using @inject decorators
    data_import_manager = injector.get(DataImportManager)
    strip_service = injector.get(StripService)
    preprocess_service = injector.get(PreprocessService)

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
    return None, final_datasets


def _create_environments_and_train(
    injector,
    final_datasets: List[PreprocessingResult],
    env_class: Type[BaseTradingEnv],
) -> Tuple[Dict[str, BaseAgent]]:
    """
    Create environments and train agents using injected services.

    Args:
        injector: The injector instance with resolved services
        final_datasets: List of preprocessing results containing final DataFrames
        env_class: The class of the trading environment to be used

    Returns:
        Dictionary of trained agents mapped by name
    """
    # Resolve required services from injector
    split_service = injector.get(SplitService)
    agent_training_service = injector.get(AgentTrainingService)

    # Extract final DataFrames from preprocessing results and split datasets
    split_datasets = []
    for preprocessing_result in final_datasets:
        # Extract the final DataFrame from the preprocessing result
        final_dataframe = preprocessing_result.final_result
        split_datasets.append(split_service.split_dataset(final_dataframe))

    # Create environments and train agents
    logger.info(
        f"Creating environments and training agents with {len(split_datasets)} datasets"
    )
    train_env, val_env, agents = agent_training_service.create_env_and_train_agents(
        split_datasets, env_class
    )
    return agents


def bootstrap_agent_training(
    env_class: Type[BaseTradingEnv],
    config_path: Optional[str] = None,
    feature_class_discovery_package: Optional[str] = None,
    feature_config_discovery_package: Optional[str] = None,
) -> None:
    """
    Bootstraps the agent training process.

    Initializes the application, preprocesses data, creates training
    and validation environments, and trains agents.

    The configuration path is determined in the following order of precedence:
    1. The `config_path` parameter, if provided.
    2. The `DRL_TRADING_CONFIG_PATH` environment variable, if set.
    3. The default path hardcoded in the modern injector container.

    Args:
        env_class: The class of the trading environment to be used.
                   Must be a subclass of BaseTradingEnv.
        config_path: Optional path to the configuration file. If None, the system
                     will attempt to use DRL_TRADING_CONFIG_PATH or the default.
    """
    # Log the intention based on parameters and environment variables
    env_var_value = os.getenv("DRL_TRADING_CONFIG_PATH")
    if config_path:
        logger.info(
            f"bootstrap_agent_training called with explicit config_path: {config_path}"
        )
    elif env_var_value:
        logger.info(
            f"bootstrap_agent_training: config_path is None, DRL_TRADING_CONFIG_PATH is set to: {env_var_value}"
        )
    else:
        logger.info(
            "bootstrap_agent_training: config_path is None and DRL_TRADING_CONFIG_PATH is not set. Container will use its default path."
        )

    # Bootstrap application with DI - _preprocess returns (None, final_datasets)
    _, final_datasets = _preprocess(
        feature_class_discovery_package, feature_config_discovery_package, config_path
    )

    # Get the injector for training services
    injector = get_trading_injector(config_path)

    # Create environments and train agents
    agents = _create_environments_and_train(injector, final_datasets, env_class)

    logger.info(f"Training completed for {len(agents)} agents")
    logger.info("Agent training bootstrap completed successfully")


def bootstrap_inference(
    config_path: Optional[str] = None,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> InferenceService:
    """
    Bootstraps the inference process for real-time trading.

    This function initializes the real-time preprocessing pipeline and inference
    service for a specific symbol, enabling real-time feature computation and
    model predictions.

    Args:
        config_path: Optional path to config file. If provided, it overrides any
                     path set by DRL_TRADING_CONFIG_PATH environment variable or default.
        symbol: Trading symbol to initialize for inference (default: "EURUSD")
        timeframe: Data timeframe for inference (default: "H1")

    Returns:
        InferenceService: Configured inference service ready for real-time predictions

    Raises:
        RuntimeError: If inference initialization fails
    """
    logger.info(f"Starting inference bootstrap for {symbol} {timeframe}")

    try:
        # Initialize logging
        configure_logging()

        # Get the trading injector with optional config path
        injector = get_trading_injector(config_path)  # Log config path if provided
        if config_path:
            logger.info(f"Using config path: {config_path}")

        # Create inference service using injected dependencies
        inference_service = injector.get(InferenceService)

        # Load historical data for the symbol to initialize preprocessing
        data_import_manager = injector.get(DataImportManager)
        symbol_data = data_import_manager.get_data()

        # Find data for the requested symbol and timeframe
        historical_data = None
        for symbol_container in symbol_data:
            if symbol_container.symbol == symbol:
                for dataset in symbol_container.datasets:
                    if dataset.timeframe == timeframe:
                        historical_data = dataset.asset_price_dataset
                        break
                break

        if historical_data is None or historical_data.empty:
            raise RuntimeError(f"No historical data found for {symbol} {timeframe}")

        # Initialize inference for the symbol
        success = inference_service.initialize_for_symbol(
            symbol=symbol, timeframe=timeframe, historical_data=historical_data
        )

        if not success:
            raise RuntimeError(
                f"Failed to initialize inference for {symbol} {timeframe}"
            )

        logger.info(f"Successfully initialized inference for {symbol} {timeframe}")
        logger.info("Inference service ready for real-time predictions")

        return inference_service

    except Exception as e:
        logger.error(f"Failed to bootstrap inference: {e}", exc_info=True)
        raise RuntimeError(f"Inference bootstrap failed: {e}") from e
