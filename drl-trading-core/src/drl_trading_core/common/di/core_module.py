"""Modern dependency injection container using injector library."""

import logging
import os
from typing import Optional, Type

from drl_trading_common.base.base_trading_env import BaseTradingEnv
from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.config.context_feature_config import ContextFeatureConfig
from drl_trading_common.config.environment_config import EnvironmentConfig
from drl_trading_common.config.feature_config import FeaturesConfig, FeatureStoreConfig
from drl_trading_common.config.local_data_import_config import LocalDataImportConfig
from drl_trading_common.config.rl_model_config import RlModelConfig
from drl_trading_common.interfaces.technical_indicator_service_interface import (
    TechnicalIndicatorFactoryInterface,
)
from feast import FeatureStore
from injector import Module, provider, singleton

from drl_trading_core.common.config.utils import parse_all_parameters
from drl_trading_core.common.data_import.base_data_import_service import (
    BaseDataImportService,
)
from drl_trading_core.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_core.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)
from drl_trading_core.preprocess.data_set_utils.context_feature_service import (
    ContextFeatureService,
    ContextFeatureServiceInterface,
)
from drl_trading_core.preprocess.data_set_utils.merge_service import (
    MergeService,
    MergeServiceInterface,
)
from drl_trading_core.preprocess.data_set_utils.split_service import (
    SplitService,
    SplitServiceInterface,
)
from drl_trading_core.preprocess.data_set_utils.strip_service import (
    StripService,
    StripServiceInterface,
)
from drl_trading_core.preprocess.feast.feast_service import (
    FeastService,
    FeastServiceInterface,
)
from drl_trading_core.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
    FeatureAggregatorInterface,
)
from drl_trading_core.preprocess.feature.feature_factory import (
    FeatureFactoryInterface,
)
from drl_trading_core.preprocess.preprocess_service import (
    PreprocessService,
    PreprocessServiceInterface,
)
from drl_trading_core.training.services.agent_training_service import (
    AgentTrainingService,
    AgentTrainingServiceInterface,
)

logger = logging.getLogger(__name__)


def _resolve_feature_store_path(
    feature_store_config: FeatureStoreConfig,
) -> Optional[str]:
    """Resolve the feature store path based on configuration."""
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


class CoreModule(Module):
    """Main application module for dependency injection.

    This module only provides configuration values and complex factory logic.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the module with optional config path."""
        self.config_path = config_path

    @provider
    @singleton
    def provide_config_path(self) -> str:
        """Provide configuration path with fallback logic."""
        if self.config_path:
            return self.config_path

        env_config_path = os.getenv("DRL_TRADING_CONFIG_PATH")
        if env_config_path:
            return env_config_path

        # Return default path if nothing else is set
        default_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "config",
            "applicationConfig.json",
        )
        return default_path

    @provider
    @singleton
    def provide_application_config(
        self, config_path: str, feature_factory: FeatureFactoryInterface
    ) -> ApplicationConfig:
        """Provide the main application configuration."""
        cfg = ConfigLoader.get_config(path=config_path)
        parse_all_parameters(cfg.features_config.feature_definitions, feature_factory)
        return cfg

    # Configuration sections - these are just data extractions
    @provider
    def provide_features_config(
        self, application_config: ApplicationConfig
    ) -> FeaturesConfig:
        """Provide features configuration."""
        return application_config.features_config

    @provider
    def provide_local_data_import_config(
        self, application_config: ApplicationConfig
    ) -> LocalDataImportConfig:
        """Provide local data import configuration."""
        return application_config.local_data_import_config

    @provider
    def provide_rl_model_config(
        self, application_config: ApplicationConfig
    ) -> RlModelConfig:
        """Provide RL model configuration."""
        return application_config.rl_model_config

    @provider
    def provide_environment_config(
        self, application_config: ApplicationConfig
    ) -> EnvironmentConfig:
        """Provide environment configuration."""
        return application_config.environment_config

    @provider
    def provide_feature_store_config(
        self, application_config: ApplicationConfig
    ) -> FeatureStoreConfig:
        """Provide feature store configuration."""
        return application_config.feature_store_config

    @provider
    def provide_context_feature_config(
        self, application_config: ApplicationConfig
    ) -> ContextFeatureConfig:
        """Provide context feature configuration."""
        return (
            application_config.context_feature_config
        )  # Complex factory logic that can't be auto-wired

    @provider
    @singleton
    def provide_feature_store(
        self, feature_store_config: FeatureStoreConfig
    ) -> Optional[FeatureStore]:
        """Provide FeatureStore instance with complex path resolution."""
        repo_path = _resolve_feature_store_path(feature_store_config)
        return FeatureStore(repo_path=repo_path) if repo_path else None

    # Service providers - these bind interfaces to concrete implementations
    @provider
    @singleton
    def provide_merge_service(self) -> MergeServiceInterface:
        """Provide MergeService implementation."""
        return MergeService()

    @provider
    @singleton
    def provide_strip_service(self) -> StripServiceInterface:
        """Provide StripService implementation."""
        return StripService()

    @provider
    @singleton
    def provide_split_service(self, config: RlModelConfig) -> SplitServiceInterface:
        """Provide SplitService implementation."""
        return SplitService(config)

    @provider
    @singleton
    def provide_context_feature_service(
        self, config: ContextFeatureConfig
    ) -> ContextFeatureServiceInterface:
        """Provide ContextFeatureService implementation."""
        return ContextFeatureService(config)

    @provider
    @singleton
    def provide_feast_service(
        self, config: FeatureStoreConfig, feature_store: Optional[FeatureStore]
    ) -> FeastServiceInterface:
        """Provide FeastService implementation."""
        return FeastService(config, feature_store)

    @provider
    @singleton
    def provide_feature_aggregator(
        self,
        config: FeaturesConfig,
        feature_factory: FeatureFactoryInterface,
        feast_service: FeastServiceInterface,
        technical_indicator_factory: TechnicalIndicatorFactoryInterface
    ) -> FeatureAggregatorInterface:
        """Provide FeatureAggregator implementation."""
        return FeatureAggregator(config, feature_factory, feast_service, technical_indicator_factory)

    @provider
    @singleton
    def provide_preprocess_service(
        self,
        features_config: FeaturesConfig,
        feature_factory: FeatureFactoryInterface,
        feature_aggregator: FeatureAggregatorInterface,
        merge_service: MergeServiceInterface,
        context_feature_service: ContextFeatureServiceInterface,
    ) -> PreprocessServiceInterface:
        """Provide PreprocessService implementation."""
        return PreprocessService(
            features_config,
            feature_factory,
            feature_aggregator,
            merge_service,
            context_feature_service,
        )

    @provider
    @singleton
    def provide_csv_data_import_service(
        self, config: LocalDataImportConfig
    ) -> BaseDataImportService:
        """Provide CSV data import service implementation."""
        return CsvDataImportService(config)

    @provider
    @singleton
    def provide_data_import_manager(
        self, import_service: BaseDataImportService
    ) -> DataImportManager:
        """Provide DataImportManager implementation."""
        return DataImportManager(import_service)

    @provider
    @singleton
    def provide_agent_training_service(
        self, config: ApplicationConfig, env_type: Type[BaseTradingEnv]
    ) -> AgentTrainingServiceInterface:
        """Provide AgentTrainingService implementation."""
        return AgentTrainingService(config, env_type)

    # Need in "drl-trading-training" or "drl-trading-inference"
    # @provider
    # @singleton
    # def provide_message_bus(self) -> TradingMessageBus:
    #     """Provide message bus based on deployment mode."""
    #     return TradingMessageBusFactory.create_message_bus()
