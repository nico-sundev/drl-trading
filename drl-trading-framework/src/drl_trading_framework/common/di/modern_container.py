"""Modern dependency injection container using injector library."""

import logging
import os
from typing import Optional

from drl_trading_common.config.application_config import ApplicationConfig
from drl_trading_common.config.config_loader import ConfigLoader
from drl_trading_common.config.context_feature_config import ContextFeatureConfig
from drl_trading_common.config.environment_config import EnvironmentConfig
from drl_trading_common.config.feature_config import FeaturesConfig, FeatureStoreConfig
from drl_trading_common.config.local_data_import_config import LocalDataImportConfig
from drl_trading_common.config.rl_model_config import RlModelConfig
from drl_trading_common.messaging import (
    DeploymentMode,
    TradingMessageBus,
    TradingMessageBusFactory,
)
from feast import FeatureStore
from injector import Injector, Module, provider, singleton

from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactoryInterface,
)
from drl_trading_framework.common.config.utils import parse_all_parameters

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


class TradingApplicationModule(Module):
    """Main application module for dependency injection.

    This module only provides configuration values and complex factory logic.
    Services are auto-wired using @inject decorators on their constructors.
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
        self, config_path: str, feature_config_factory: FeatureConfigFactoryInterface
    ) -> ApplicationConfig:
        """Provide the main application configuration."""
        cfg = ConfigLoader.get_config(path=config_path)
        parse_all_parameters(
            cfg.features_config.feature_definitions, feature_config_factory
        )
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
        return application_config.context_feature_config

    # Complex factory logic that can't be auto-wired
    @provider
    @singleton
    def provide_feature_store(
        self, feature_store_config: FeatureStoreConfig
    ) -> Optional[FeatureStore]:
        """Provide FeatureStore instance with complex path resolution."""
        repo_path = _resolve_feature_store_path(feature_store_config)
        return FeatureStore(repo_path=repo_path) if repo_path else None

    @provider
    @singleton
    def provide_deployment_mode(self) -> DeploymentMode:
        """Provide deployment mode from environment."""
        mode_str = os.getenv("DEPLOYMENT_MODE", "training")
        return DeploymentMode(mode_str)

    @provider
    @singleton
    def provide_message_bus(self, deployment_mode: DeploymentMode) -> TradingMessageBus:
        """Provide message bus based on deployment mode."""
        return TradingMessageBusFactory.create_message_bus(deployment_mode)


# Global injector instance
_trading_injector: Optional[Injector] = None


def get_trading_injector(config_path: Optional[str] = None) -> Injector:
    """Get or create the global trading injector."""
    global _trading_injector

    if _trading_injector is None:
        module = TradingApplicationModule(config_path)
        _trading_injector = Injector([module])
        logger.info("Trading injector initialized")

    return _trading_injector


def reset_trading_injector():
    """Reset the global injector (useful for testing)."""
    global _trading_injector
    _trading_injector = None


def get_service(service_class, config_path: Optional[str] = None):
    """Convenience function to get a service from the injector."""
    injector = get_trading_injector(config_path)
    return injector.get(service_class)
