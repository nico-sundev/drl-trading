import logging
import os
from typing import Dict, List, Optional, Tuple, Type

from drl_trading_common.config.logging_config import configure_logging
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_trading_framework.common.agents.base_agent import BaseAgent
from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)
from drl_trading_framework.common.di.containers import ApplicationContainer
from drl_trading_framework.common.gym import T
from drl_trading_framework.common.model.preprocessing_result import PreprocessingResult
from drl_trading_framework.inference.inference_service import InferenceService
from drl_trading_framework.preprocess.feature.feature_class_factory import (
    FeatureClassFactory,
)

logger = logging.getLogger(__name__)


# Updated _preprocess to accept factory interfaces
def _preprocess(
    feature_class_discovery_package: Optional[str] = None,
    feature_config_discovery_package: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[ApplicationContainer, List[PreprocessingResult]]:
    """Bootstrap the application using dependency injection.

    This function initializes the application container, configures logging,
    and processes data through the pipeline.

    Args:
        config_path: Optional path to config file. If provided, it overrides any
                     path set by DRL_TRADING_CONFIG_PATH environment variable or default.

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

    # Determine effective config path and log it
    # Priority:
    # 1. Explicit config_path parameter
    # 2. DRL_TRADING_CONFIG_PATH environment variable (handled by container's default)
    # 3. Container's hardcoded DEFAULT_CONFIG_PATH (handled by container's default)

    env_var_value = os.getenv("DRL_TRADING_CONFIG_PATH")
    initial_loaded_path = (
        container.config_path_cfg()
    )  # Path loaded by container (env var or default)

    if config_path:
        if config_path != initial_loaded_path:
            logger.info(
                f"Explicit config_path parameter '{config_path}' provided, overriding container's initial path '{initial_loaded_path}'."
            )
            container.config_path_cfg.override(config_path)
            logger.info(
                f"Container config path now set to: {container.config_path_cfg()}"
            )
        else:
            logger.info(
                f"Explicit config_path parameter '{config_path}' matches container's initial path (likely from env var or default). Using: {config_path}"
            )
    elif env_var_value:
        logger.info(
            f"Using config path from DRL_TRADING_CONFIG_PATH environment variable: {env_var_value}"
        )
    else:
        logger.info(
            f"No explicit config_path parameter and DRL_TRADING_CONFIG_PATH not set. Using default config path: {initial_loaded_path}"
        )

    logger.info("DI container configured")

    class_factory = FeatureClassFactory()
    if feature_class_discovery_package:
        logger.info(
            f"Discovering feature classes from package: {feature_class_discovery_package}"
        )
        class_factory.discover_feature_classes(feature_class_discovery_package)

    config_factory = FeatureConfigFactory()
    if feature_config_discovery_package:
        logger.info(
            f"Discovering feature config classes from package: {feature_config_discovery_package}"
        )
        config_factory.discover_config_classes(feature_config_discovery_package)

    # Inject the pre-initialized feature config factory and feature class factory into the container
    container.feature_config_factory.override(config_factory)
    container.feature_class_registry.override(class_factory)

    # Parse feature parameters using the injected feature config factory
    features_config = container.features_config()
    features_config.parse_all_parameters(container.feature_config_factory())
    logger.info("Feature parameters parsed successfully")

    # Remove discovery logic as it is handled externally
    logger.info("Feature configuration and class factories injected into the container")

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


def _create_environments_and_train(
    container: ApplicationContainer,
    final_datasets: List[PreprocessingResult],
    env_class: Type[T],
) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
    """
    Create environments and train agents using injected services.

    Args:
        container: Application container with resolved services
        final_datasets: List of preprocessing results containing final DataFrames

    Returns:
        Tuple containing:
        - Training environment (DummyVecEnv)
        - Validation environment (DummyVecEnv)
        - Dictionary of trained agents mapped by name
    """
    # Resolve required services
    split_service = container.split_service()
    agent_training_service = container.agent_training_service()

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
    return agent_training_service.create_env_and_train_agents(split_datasets, env_class)


def bootstrap_agent_training(
    env_class: Type[T],
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
    3. The default path hardcoded in the ApplicationContainer.

    Args:
        env_class: The class of the trading environment to be used.
                   Must be a subclass of BaseTradingEnv.
        config_path: Optional path to the configuration file. If None, the system
                     will attempt to use DRL_TRADING_CONFIG_PATH or the default.
    Raises:
        ValueError: If no configuration path can be resolved (neither parameter,
                    nor environment variable is set, and the default path is
                    considered invalid or not explicitly desired for critical operations).
                    Currently, this check is implicitly handled by ConfigLoader;
                    this docstring reflects the intent.
                    A direct check for a usable path is more robust.
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
        # Consider adding a check here if the default path is acceptable or exists,
        # if not providing any config explicitly is an error condition.
        # For now, we let _preprocess and the container handle it.
        # The original check was:
        # if config_path is None:
        # raise ValueError("Config path must be provided. Please specify a valid path.")
        # This is now relaxed as the container handles env var and default.
        # However, if relying on the hardcoded default is undesirable, a check could be:
        # if not config_path and not env_var_value and ApplicationContainer().config_path_cfg() == ApplicationContainer.DEFAULT_CONFIG_PATH:
        #     # This means we are falling back to the hardcoded default.
        #     # Depending on requirements, this might warrant a warning or error.
        #     pass # Current behavior: allow fallback to default.

    # Bootstrap application with DI.
    # _preprocess will use config_path if provided, otherwise container defaults (env_var or hardcoded default)
    container, final_datasets = _preprocess(
        feature_class_discovery_package, feature_config_discovery_package, config_path
    )

    # Create environments and train agents
    train_env, val_env, agents = _create_environments_and_train(
        container, final_datasets, env_class
    )


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
        RuntimeError: If inference initialization fails"""
    logger.info(f"Starting inference bootstrap for {symbol} {timeframe}")

    try:
        # Initialize logging
        configure_logging()

        # Create and configure the DI container
        container = ApplicationContainer()

        # Set config path if provided
        if config_path:
            container.config_path_cfg.override(config_path)
            logger.info(f"Using config path: {config_path}")

        # Get real-time preprocessing service
        real_time_preprocess_service = container.real_time_preprocess_service()

        # Create inference service
        inference_service = InferenceService(
            real_time_preprocess_service=real_time_preprocess_service
        )

        # Load historical data for the symbol to initialize preprocessing
        data_import_manager = container.data_import_manager()
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
