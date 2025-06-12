from typing import Type, TypeVar

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from .feature_config import FeaturesConfig

# Generic type variable bounded by BaseApplicationConfig
T = TypeVar('T', bound=BaseApplicationConfig)


class ConfigLoader:
    @staticmethod
    def feature_config(path: str) -> FeaturesConfig:
        with open(path) as f:
            return FeaturesConfig.model_validate_json(f.read())

    @staticmethod
    def get_config(config_class: Type[T], path: str) -> T:
        """
        Generic method to parse any configuration object that extends BaseApplicationConfig.

        Args:
            config_class: The concrete configuration class to instantiate
            path: Path to the JSON configuration file

        Returns:
            Instance of the specified configuration class

        Example:
            app_config = ConfigLoader.get_config(ApplicationConfig, "path/to/config.json")
            custom_config = ConfigLoader.get_config(CustomConfig, "path/to/custom.json")
        """
        with open(path) as f:
            return config_class.model_validate_json(f.read())
