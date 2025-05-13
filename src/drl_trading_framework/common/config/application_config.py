from drl_trading_framework.common.config.base_schema import BaseSchema
from drl_trading_framework.common.config.environment_config import EnvironmentConfig
from drl_trading_framework.common.config.feature_config import (
    FeaturesConfig,
    FeatureStoreConfig,
)
from drl_trading_framework.common.config.local_data_import_config import (
    LocalDataImportConfig,
)
from drl_trading_framework.common.config.rl_model_config import RlModelConfig


class ApplicationConfig(BaseSchema):
    features_config: FeaturesConfig
    local_data_import_config: LocalDataImportConfig
    rl_model_config: RlModelConfig
    environment_config: EnvironmentConfig
    feature_store_config: FeatureStoreConfig
