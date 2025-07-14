"""Modern dependency injection container using injector library."""

import logging
import os
from typing import Optional

from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.config.context_feature_config import ContextFeatureConfig
from drl_trading_common.config.environment_config import EnvironmentConfig
from drl_trading_common.config.feature_config import FeaturesConfig, FeatureStoreConfig
from drl_trading_common.config.feature_config_repo import (
    FeatureConfigPostgresRepo,
    IFeatureConfigRepository,
)
from drl_trading_common.config.local_data_import_config import LocalDataImportConfig
from drl_trading_common.config.rl_model_config import RlModelConfig
from injector import Module, provider, singleton

from drl_trading_core.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_core.common.data_import.data_import_strategy_factory import (
    DataImportStrategyFactory,
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
from drl_trading_core.preprocess.feature.feature_manager import FeatureManager
from drl_trading_core.preprocess.feature_store.provider.feast_provider import (
    FeastProvider,
)
from drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper import (
    FeatureStoreWrapper,
)
from drl_trading_core.preprocess.feature_store.repository.feature_store_fetch_repo import (
    FeatureStoreFetchRepository,
    IFeatureStoreFetchRepository,
)
from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
    FeatureStoreSaveRepository,
    IFeatureStoreSaveRepository,
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


class CoreModule(Module):
    """Main application module for dependency injection.

    This module provides configuration values, complex factory logic, and interface bindings.
    Services with @inject decorators are auto-wired through interface bindings.
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
        self, config_path: str
    ) -> ApplicationConfig:
        """Provide the main application configuration."""
        cfg = ConfigLoader.get_config(ApplicationConfig, config_path)
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
        )
    def configure(self, binder) -> None:
        """Configure interface bindings for auto-wiring services with @inject decorators."""
        from drl_trading_core.preprocess.compute.computing_service import (
            FeatureComputingService,
            IFeatureComputer,
        )
        from drl_trading_core.preprocess.feature_store.mapper.feature_view_name_mapper import (
            FeatureViewNameMapper,
        )
        from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo import (
            OfflineFeatureLocalRepo,
        )
        from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_repo_interface import (
            IOfflineFeatureRepository,
        )

        # Auto-wire services that use @inject decorator
        binder.bind(DataImportManager, to=DataImportManager, scope=singleton)
        binder.bind(MergeServiceInterface, to=MergeService, scope=singleton)
        binder.bind(StripServiceInterface, to=StripService, scope=singleton)
        binder.bind(SplitServiceInterface, to=SplitService, scope=singleton)
        binder.bind(
            IFeatureComputer,
            to=FeatureComputingService,
            scope=singleton,
        )
        binder.bind(
            FeatureStoreWrapper, to=FeatureStoreWrapper, scope=singleton
        )
        binder.bind(
            IFeatureConfigRepository, to=FeatureConfigPostgresRepo, scope=singleton
        )
        binder.bind(FeastProvider, to=FeastProvider, scope=singleton)
        binder.bind(
            IOfflineFeatureRepository, to=OfflineFeatureLocalRepo, scope=singleton
        )
        binder.bind(
            FeatureViewNameMapper, to=FeatureViewNameMapper, scope=singleton
        )
        binder.bind(
            IFeatureStoreSaveRepository, to=FeatureStoreSaveRepository, scope=singleton
        )
        binder.bind(
            IFeatureStoreFetchRepository, to=FeatureStoreFetchRepository, scope=singleton
        )
        binder.bind(FeatureManager, to=FeatureManager, scope=singleton)
        binder.bind(PreprocessServiceInterface, to=PreprocessService, scope=singleton)
        binder.bind(
            DataImportStrategyFactory, to=DataImportStrategyFactory, scope=singleton
        )
        binder.bind(
            AgentTrainingServiceInterface, to=AgentTrainingService, scope=singleton
        )
