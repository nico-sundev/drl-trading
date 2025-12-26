
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.context_feature_config import ContextFeatureConfig
from drl_trading_common.config.environment_config import EnvironmentConfig
from drl_trading_common.config.feature_config import FeaturesConfig


class ApplicationConfig(BaseApplicationConfig):
    features_config: FeaturesConfig
    environment_config: EnvironmentConfig
    context_feature_config: ContextFeatureConfig
