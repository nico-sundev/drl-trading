from ..base.base_schema import BaseSchema
from .context_feature_config import ContextFeatureConfig
from .environment_config import EnvironmentConfig
from .feature_config import FeaturesConfig, FeatureStoreConfig
from .local_data_import_config import LocalDataImportConfig
from .rl_model_config import RlModelConfig


class ApplicationConfig(BaseSchema):
    features_config: FeaturesConfig
    local_data_import_config: LocalDataImportConfig
    rl_model_config: RlModelConfig
    environment_config: EnvironmentConfig
    feature_store_config: FeatureStoreConfig
    context_feature_config: ContextFeatureConfig
