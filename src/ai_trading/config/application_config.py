from ai_trading.config.base_schema import BaseSchema
from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.config.local_data_import_config import LocalDataImportConfig
from ai_trading.config.model_config import RlModelConfig


class ApplicationConfig(BaseSchema):
    features_config: FeaturesConfig
    local_data_import_config: LocalDataImportConfig
    rl_model_config: RlModelConfig
    environment_config: EnvironmentConfig