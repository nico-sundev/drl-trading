"""Dependency injection container configuration using dependency-injector."""

import logging
import os

from dependency_injector import containers, providers

from drl_trading_framework.common.config.config_loader import ConfigLoader
from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)
from drl_trading_framework.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_framework.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
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


class ApplicationContainer(containers.DeclarativeContainer):
    """Application container for dependency injection.

    This container manages the lifecycle of application services and
    their dependencies. It maintains the ApplicationConfig and its child
    config classes as separate injectable components.
    """

    # Register full ApplicationConfig as a singleton
    application_config = providers.Singleton(
        ConfigLoader.get_config,
        path=os.path.join(
            os.path.dirname(__file__), "../../../configs/applicationConfig.json"
        ),
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
        atr_period=14,
    )

    # Data import services - using directly injected config sections
    csv_data_import_service = providers.Singleton(
        CsvDataImportService,
        config=local_data_import_config,
    )

    data_import_manager = providers.Singleton(
        DataImportManager,
        import_service=csv_data_import_service,
    )

    # Feast and feature related services - direct config injection
    feast_service = providers.Singleton(
        FeastService,
        config=feature_store_config,
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
