"""Service-specific configuration for preprocess service."""
from datetime import datetime
from pydantic import Field
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.feature_config import FeatureStoreConfig
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


class ResampleConfig(BaseSchema):
    """Configuration for market data resampling operations."""
    # Historical data limits to prevent memory issues
    historical_start_date: datetime = datetime(2020, 1, 1)
    max_batch_size: int = 100000  # Maximum records to process in memory

    # Performance settings
    progress_log_interval: int = 10000  # Log progress every N records
    enable_incomplete_candle_publishing: bool = True  # Emit incomplete final candles (useful for real-time scenarios)

    # Memory management
    chunk_size: int = 50000  # Process data in chunks of this size
    memory_warning_threshold_mb: int = 1000  # Warn if memory usage exceeds this

    # Pagination settings for stateful processing
    pagination_limit: int = 10000  # Number of records to fetch per page
    max_memory_usage_mb: int = 512  # Maximum memory usage before triggering cleanup

    # State persistence configuration
    state_persistence_enabled: bool = True
    state_file_path: str = "state/resampling_context.json"  # Path to store ResamplingContext state
    state_backup_interval: int = 1000  # Save state every N processed records
    auto_cleanup_inactive_symbols: bool = True  # Automatically clean up inactive symbols
    inactive_symbol_threshold_hours: int = 24  # Hours after which symbols are considered inactive


class PreprocessConfig(BaseApplicationConfig):
    """Configuration for preprocess service - focused on data processing."""
    app_name: str = "drl-trading-preprocess"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    # T005 logging configuration for ServiceLogger
    logging: ServiceLoggingConfig = Field(default_factory=ServiceLoggingConfig)
    feature_store_config: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    resample_config: ResampleConfig = Field(default_factory=ResampleConfig)
