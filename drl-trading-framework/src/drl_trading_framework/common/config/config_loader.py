from drl_trading_framework.common.config.application_config import ApplicationConfig
from drl_trading_framework.common.config.feature_config import FeaturesConfig


class ConfigLoader:
    @staticmethod
    def feature_config(path: str) -> FeaturesConfig:
        with open(path) as f:
            return FeaturesConfig.model_validate_json(f.read())

    @staticmethod
    def get_config(path: str) -> ApplicationConfig:
        with open(path) as f:
            return ApplicationConfig.model_validate_json(f.read())
