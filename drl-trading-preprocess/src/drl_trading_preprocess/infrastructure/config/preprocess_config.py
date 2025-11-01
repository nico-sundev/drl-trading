"""Service-specific configuration for preprocess service."""
from datetime import datetime
from typing import List
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


class TopicSubscription(BaseSchema):
    """Configuration for a single Kafka topic subscription.
    
    Maps a Kafka topic to a handler identifier that will be looked up
    in the DI container's handler registry.
    
    Attributes:
        topic: Name of the Kafka topic to subscribe to.
        handler_id: Identifier for the handler function in the DI registry.
            This allows topic names to be configured in YAML while handler
            implementations remain in code.
    """
    topic: str = Field(..., description="Kafka topic name")
    handler_id: str = Field(..., description="Handler identifier for DI lookup")


class KafkaConsumerConfig(BaseSchema):
    """Service-specific Kafka consumer configuration.
    
    This configuration extends the infrastructure-level KafkaConnectionConfig
    with service-specific consumer settings and topic subscriptions.
    
    Design: Configuration-driven handler mapping
    - Topic names live in YAML (infrastructure concern)
    - Handler IDs link to implementations in DI module (application concern)
    - Clean separation between deployment config and code
    
    Attributes:
        consumer_group_id: Kafka consumer group ID for this service.
            Should be unique per service to enable independent consumption.
        topic_subscriptions: List of topics this service consumes from.
            Each subscription maps to a handler in the DI registry.
    """
    consumer_group_id: str = Field(
        default="drl-trading-preprocess-group",
        description="Kafka consumer group ID for this service"
    )
    topic_subscriptions: List[TopicSubscription] = Field(
        default_factory=list,
        description="List of topic-to-handler mappings"
    )


class KafkaTopicConfig(BaseSchema):
    """Configuration for a single Kafka producer topic.
    
    Maps a topic name to its associated retry configuration key.
    This allows per-topic resilience behavior (e.g., critical data
    gets more aggressive retries than best-effort notifications).
    
    Attributes:
        topic: Name of the Kafka topic to publish to.
        retry_config_key: Key to look up retry configuration from
            infrastructure.resilience.retry_configs. Should reference
            a constant from resilience_constants.py for type safety.
    """
    topic: str = Field(..., description="Kafka topic name")
    retry_config_key: str = Field(
        ...,
        description="Retry configuration key from infrastructure.resilience.retry_configs"
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
    feature_store_config: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    resample_config: ResampleConfig = Field(default_factory=ResampleConfig)
    kafka_consumers: KafkaConsumerConfig = Field(default_factory=KafkaConsumerConfig)
    kafka_topics: KafkaTopicsConfig | None = Field(
        default=None,
        description="Kafka producer topic configurations"
    )
