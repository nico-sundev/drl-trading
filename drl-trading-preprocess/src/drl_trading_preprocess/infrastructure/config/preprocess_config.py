"""Service-specific configuration for preprocess service."""
from pydantic import Field
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.infrastructure_config import InfrastructureConfig
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig


class DataSourceConfig(BaseSchema):
    """Data source configuration for preprocessing."""
    input_path: str = "data/raw/"
    supported_formats: list[str] = ["csv", "parquet", "json"]
    batch_size: int = 1000


class FeatureEngineeringConfig(BaseSchema):
    """Feature engineering configuration."""
    enabled_features: list[str] = ["technical_indicators", "market_data", "sentiment"]
    lookback_period: int = 30
    scaling_method: str = "standard"  # standard | minmax | robust


class OutputConfig(BaseSchema):
    """Output configuration for processed data."""
    output_path: str = "data/processed/"
    format: str = "parquet"
    compression: str = "snappy"
    validation_enabled: bool = True


class PreprocessConfig(BaseApplicationConfig):
    """Configuration for preprocess service - focused on data processing."""
    app_name: str = "drl-trading-preprocess"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    # T005 logging configuration for ServiceLogger
    logging: ServiceLoggingConfig = Field(default_factory=ServiceLoggingConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
