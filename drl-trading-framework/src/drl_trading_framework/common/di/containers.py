"""Dependency injection container configuration using dependency-injector."""

import logging
import os
from typing import Optional

from dependency_injector import containers, providers
from feast import FeatureStore

from drl_trading_framework.common.config.config_loader import ConfigLoader
from drl_trading_framework.common.config.feature_config import FeatureStoreConfig
from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)
from drl_trading_framework.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_framework.common.data_import.data_import_strategy_factory import (
    DataImportStrategyFactory,
)
from drl_trading_framework.preprocess.data_set_utils.context_feature_service import (
    ContextFeatureService,
)
from drl_trading_framework.preprocess.data_set_utils.merge_service import MergeService
from drl_trading_framework.preprocess.data_set_utils.split_service import SplitService
from drl_trading_framework.preprocess.data_set_utils.strip_service import StripService
from drl_trading_framework.preprocess.feast.feast_service import FeastService
from drl_trading_framework.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
)
from drl_trading_framework.preprocess.feature.feature_class_registry import (
    FeatureClassRegistry,
)
from drl_trading_framework.preprocess.preprocess_service import PreprocessService
from drl_trading_framework.training.services.agent_training_service import (
    AgentTrainingService,
)

logger = logging.getLogger(__name__)


def _resolve_feature_store_path(
    feature_store_config: FeatureStoreConfig,
) -> Optional[str]:

    if not feature_store_config.enabled:
        return None

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if not os.path.isabs(feature_store_config.repo_path):
        abs_file_path = os.path.join(project_root, feature_store_config.repo_path)
    else:
        abs_file_path = feature_store_config.repo_path
    return abs_file_path


class ApplicationContainer(containers.DeclarativeContainer):
    """Application container for dependency injection.

    This container manages the lifecycle of application services and
    their dependencies. It maintains the ApplicationConfig and its child
    config classes as separate injectable components.
    """

    # Configuration provider for the application config path
    # It will first try to get the path from the DRL_TRADING_CONFIG_PATH environment variable.
    # If the environment variable is not set, it will use the DEFAULT_CONFIG_PATH.
    config_path_cfg = providers.Configuration(
        name="application_config_path", default=os.getenv("DRL_TRADING_CONFIG_PATH")
    )

    # Register full ApplicationConfig as a singleton
    application_config = providers.Singleton(
        ConfigLoader.get_config,
        path=config_path_cfg,  # Use the configuration provider for the path
    )

    # Register individual config sections as separately injectable components
    # These use attribute getters on the main config to avoid duplication
    features_config = providers.Callable(
        lambda config: config.features_config, config=application_config
    )

    local_data_import_config = providers.Callable(
        lambda config: config.local_data_import_config, config=application_config
    )

    rl_model_config = providers.Callable(
        lambda config: config.rl_model_config, config=application_config
    )

    environment_config = providers.Callable(
        lambda config: config.environment_config, config=application_config
    )

    feature_store_config = providers.Callable(
        lambda config: config.feature_store_config, config=application_config
    )

    context_feature_config = providers.Callable(
        lambda config: config.context_feature_config, config=application_config
    )

    # Feature configuration factory - replaces FeatureConfigRegistry singleton
    feature_config_factory = providers.Singleton(
        FeatureConfigFactory,
    )

    # Core utilities and stateless services
    feature_class_registry = providers.Singleton(FeatureClassRegistry)
    merge_service = providers.Singleton(MergeService)
    strip_service = providers.Singleton(StripService)
    context_feature_service = providers.Singleton(
        ContextFeatureService,
        context_feature_config,
        atr_period=14,
    )  # Data import strategy factory
    data_import_strategy_factory = providers.Singleton(
        DataImportStrategyFactory,
    )

    # Data import service - created dynamically based on strategy
    data_import_service = providers.Singleton(
        lambda factory, config: factory.create_import_service(config),
        factory=data_import_strategy_factory,
        config=local_data_import_config,
    )

    data_import_manager = providers.Singleton(
        DataImportManager,
        import_service=data_import_service,
    )

    # Backward compatibility - alias for existing code
    csv_data_import_service = data_import_service

    feature_store_path = providers.Callable(
        _resolve_feature_store_path, feature_store_config
    )

    # Only create a FeatureStore instance if a valid path is provided, else return None
    feature_store = providers.Singleton(
        lambda repo_path: FeatureStore(repo_path=repo_path) if repo_path else None,
        feature_store_path,
    )

    # Feast and feature related services - direct config injection
    feast_service = providers.Singleton(
        FeastService, config=feature_store_config, feature_store=feature_store
    )

    feature_aggregator = providers.Singleton(
        FeatureAggregator,
        config=features_config,
        class_registry=feature_class_registry,
        feast_service=feast_service,
    )

    # Processing services - direct config injection
    preprocess_service = providers.Singleton(
        PreprocessService,
        features_config=features_config,
        feature_class_registry=feature_class_registry,
        feature_aggregator=feature_aggregator,
        merge_service=merge_service,
        context_feature_service=context_feature_service,
    )

    # Split service - direct config injection
    split_service = providers.Singleton(
        SplitService,
        config=rl_model_config,
    )

    # Training service - uses full application config
    agent_training_service = providers.Singleton(
        AgentTrainingService,
        config=application_config,
    )
