"""Service-specific configuration for inference service."""
from pydantic import Field
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.infrastructure_config import InfrastructureConfig


class ModelConfig(BaseSchema):
    """Model loading and inference configuration."""
    model_path: str = "models/"
    model_name: str = "default_model"
    device: str = "cpu"
    batch_size: int = 32


class FeatureStoreConfig(BaseSchema):
    """Feature store configuration."""
    provider: str = "local"  # local | feast | custom
    config_path: str = "feature_store.yaml"


class PredictionConfig(BaseSchema):
    """Prediction service configuration."""
    endpoint: str = "/predict"
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30


class InferenceConfig(BaseApplicationConfig):
    """Configuration for inference service - only what it needs."""
    app_name: str = "drl-trading-inference"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
