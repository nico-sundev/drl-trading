from drl_trading_common.base.base_application_config import BaseApplicationConfig
from .context_feature_config import ContextFeatureConfig
from .environment_config import EnvironmentConfig
from .feature_config import FeaturesConfig, FeatureStoreConfig
from .local_data_import_config import LocalDataImportConfig
from .rl_model_config import RlModelConfig


class ApplicationConfig(BaseApplicationConfig):
    features_config: FeaturesConfig
    local_data_import_config: LocalDataImportConfig
    rl_model_config: RlModelConfig
    environment_config: EnvironmentConfig
    feature_store_config: FeatureStoreConfig
    context_feature_config: ContextFeatureConfig
