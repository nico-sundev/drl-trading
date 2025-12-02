"""Service-specific configuration for preprocess service."""
from datetime import datetime

from pydantic import Field

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.dask_config import DaskConfig
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig
from drl_trading_common.config.kafka_config import KafkaConsumerConfig, KafkaTopicConfig
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig
from drl_trading_common.config.validators import StrictAfterMergeSchema


class FeatureComputationConfig(StrictAfterMergeSchema):
    """Configuration for feature computation operations.

    Fields are optional to allow YAML merging (base + stage-specific),
    but validation ensures all are present after merge.
    """
    # Indicator warmup settings
    warmup_candles: int | None = None  # Number of candles to use for indicator warmup


class ResampleConfig(StrictAfterMergeSchema):
    """Configuration for market data resampling operations.

    Fields are optional to allow YAML merging (base + stage-specific),
    but validation ensures all are present after merge.
    """
    # Historical data limits to prevent memory issues
    historical_start_date: datetime | None = None
    max_batch_size: int | None = None

    # Performance settings
    progress_log_interval: int | None = None
    enable_incomplete_candle_publishing: bool | None = None

    # Memory management
    chunk_size: int | None = None
    memory_warning_threshold_mb: int | None = None

    # Pagination settings for stateful processing
    pagination_limit: int | None = None
    max_memory_usage_mb: int | None = None

    # State persistence configuration
    state_persistence_enabled: bool | None = None
    state_file_path: str | None = None
    state_backup_interval: int | None = None
    auto_cleanup_inactive_symbols: bool | None = None
    inactive_symbol_threshold_hours: int | None = None


class DaskConfigs(BaseSchema):
    """Collection of Dask configurations for different use cases.

    Similar to KafkaTopicsConfig, this allows different parts of the service
    to use appropriately tuned Dask configurations based on workload characteristics.

    Use cases:
    - coverage_analysis: I/O-bound operations (DB queries, Feast fetches) across 1-4 timeframes
    - feature_computation: CPU-bound operations (computing hundreds of features) with parallelization
      across both features and timeframes

    Attributes:
        coverage_analysis: Dask config for I/O-bound coverage analysis operations
        feature_computation: Dask config for CPU-bound feature computation operations
    """
    coverage_analysis: DaskConfig = Field(
        default_factory=lambda: DaskConfig(
            scheduler="threads",  # Temporarily use threads to debug pickling issue
            num_workers=None,  # Auto-detect for I/O
            threads_per_worker=1,
        ),
        description="Dask config for I/O-bound coverage analysis (DB/Feast queries)"
    )
    feature_computation: DaskConfig = Field(
        default_factory=lambda: DaskConfig(
            scheduler="threads",  # Temporarily use threads to debug pickling issue
            num_workers=4,  # Explicit count for predictable resource usage
            threads_per_worker=1,
        ),
        description="Dask config for CPU-bound feature computation"
    )


class KafkaTopicsConfig(BaseSchema):
    """Configuration for all Kafka producer topics in the preprocess service.

    This maps use cases to topic configurations, allowing each producer
    to have its own topic name and retry behavior.

    Attributes:
        resampled_data: Topic for publishing resampled market data.
        preprocessing_completed: Topic for preprocessing completion events.
        preprocessing_error: Dead letter queue for preprocessing errors.
    """
    resampled_data: KafkaTopicConfig = Field(
        ...,
        description="Topic for publishing resampled market data"
    )
    preprocessing_completed: KafkaTopicConfig = Field(
        ...,
        description="Topic for preprocessing completion events"
    )
    preprocessing_error: KafkaTopicConfig = Field(
        ...,
        description="Dead letter queue for preprocessing errors"
    )


class PreprocessConfig(BaseApplicationConfig):
    """Configuration for preprocess service - focused on data processing."""
    app_name: str = "drl-trading-preprocess"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    # T005 logging configuration for ServiceLogger
    logging: ServiceLoggingConfig = Field(default_factory=ServiceLoggingConfig)
    feature_store_config: FeatureStoreConfig | None = None  # Stage-specific override
    feature_computation_config: FeatureComputationConfig  # Required - must be in YAML
    resample_config: ResampleConfig  # Required - must be in YAML
    dask_configs: DaskConfigs = Field(default_factory=DaskConfigs)
    kafka_consumers: KafkaConsumerConfig | None = None  # Stage-specific override
    kafka_topics: KafkaTopicsConfig | None = Field(
        default=None,
        description="Kafka producer topic configurations"
    )
