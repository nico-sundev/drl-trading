
"""Kafka configuration for messaging infrastructure."""
from typing import Optional, Dict, Any, List

from pydantic import Field

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.messaging.kafka_constants import (
    KAFKA_CONFIG_BOOTSTRAP_SERVERS,
    KAFKA_CONFIG_GROUP_ID,
    KAFKA_CONFIG_SECURITY_PROTOCOL,
    KAFKA_CONFIG_SASL_MECHANISM,
    KAFKA_CONFIG_SASL_USERNAME,
    KAFKA_CONFIG_SASL_PASSWORD,
    KAFKA_CONFIG_ACKS,
    KAFKA_CONFIG_RETRIES,
    KAFKA_CONFIG_COMPRESSION_TYPE,
    KAFKA_CONFIG_ENABLE_AUTO_COMMIT,
    KAFKA_CONFIG_AUTO_OFFSET_RESET,
    KAFKA_CONFIG_SESSION_TIMEOUT_MS,
    KAFKA_CONFIG_MAX_POLL_INTERVAL_MS,
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_SECURITY_PROTOCOL,
    DEFAULT_CONSUMER_AUTO_OFFSET_RESET,
    DEFAULT_CONSUMER_ENABLE_AUTO_COMMIT,
    DEFAULT_CONSUMER_SESSION_TIMEOUT_MS,
    DEFAULT_CONSUMER_MAX_POLL_INTERVAL_MS,
    DEFAULT_PRODUCER_ACKS,
    DEFAULT_PRODUCER_RETRIES,
    DEFAULT_PRODUCER_COMPRESSION_TYPE,
    SECURITY_PROTOCOL_SASL_PLAINTEXT,
    SECURITY_PROTOCOL_SASL_SSL,
)


class KafkaConnectionConfig(BaseSchema):
    """Infrastructure-level Kafka configuration.

    Contains connection settings and security configuration that apply
    across all services. This is the foundational Kafka configuration
    that lives in the infrastructure section of service configs.

    Service-specific settings (consumer groups, topic subscriptions)
    should be defined in service-level configuration classes.

    Attributes:
        bootstrap_servers: Comma-separated list of Kafka broker addresses.
        security_protocol: Protocol for broker communication (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL).
        sasl_mechanism: SASL authentication mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI).
        sasl_username: Username for SASL authentication.
        sasl_password: Password for SASL authentication.
        producer_acks: Producer acknowledgment mode (all, 0, 1).
        producer_retries: Number of retries for failed producer sends.
        producer_compression_type: Compression algorithm for messages (none, gzip, snappy, lz4, zstd).
        consumer_auto_offset_reset: Reset strategy when no offset exists (earliest, latest, none).
        consumer_enable_auto_commit: Whether to auto-commit offsets (False recommended for reliability).
        consumer_session_timeout_ms: Consumer session timeout in milliseconds.
        consumer_max_poll_interval_ms: Max time between poll() calls before rebalancing.
    """

    # Connection settings
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS

    # Security settings
    security_protocol: str = DEFAULT_SECURITY_PROTOCOL
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    # Producer defaults
    producer_acks: str = DEFAULT_PRODUCER_ACKS
    producer_retries: int = DEFAULT_PRODUCER_RETRIES
    producer_compression_type: str = DEFAULT_PRODUCER_COMPRESSION_TYPE

    # Consumer defaults
    consumer_auto_offset_reset: str = DEFAULT_CONSUMER_AUTO_OFFSET_RESET
    consumer_enable_auto_commit: bool = DEFAULT_CONSUMER_ENABLE_AUTO_COMMIT
    consumer_session_timeout_ms: int = DEFAULT_CONSUMER_SESSION_TIMEOUT_MS
    consumer_max_poll_interval_ms: int = DEFAULT_CONSUMER_MAX_POLL_INTERVAL_MS

    def get_producer_config(self, **overrides: Any) -> Dict[str, Any]:
        """Build confluent-kafka producer configuration dictionary.

        Args:
            **overrides: Additional or override configuration parameters.

        Returns:
            Configuration dict compatible with confluent-kafka Producer.

        Example:
            ```python
            kafka_config = KafkaConnectionConfig(bootstrap_servers="kafka:9092")
            producer_config = kafka_config.get_producer_config(
                linger_ms=10,
                batch_size=16384
            )
            producer = Producer(producer_config)
            ```
        """
        config = {
            KAFKA_CONFIG_BOOTSTRAP_SERVERS: self.bootstrap_servers,
            KAFKA_CONFIG_SECURITY_PROTOCOL: self.security_protocol,
            KAFKA_CONFIG_ACKS: self.producer_acks,
            KAFKA_CONFIG_RETRIES: self.producer_retries,
            KAFKA_CONFIG_COMPRESSION_TYPE: self.producer_compression_type,
        }

        # Add SASL authentication if required
        if self.security_protocol in [SECURITY_PROTOCOL_SASL_PLAINTEXT, SECURITY_PROTOCOL_SASL_SSL]:
            if self.sasl_mechanism:
                config[KAFKA_CONFIG_SASL_MECHANISM] = self.sasl_mechanism
            if self.sasl_username:
                config[KAFKA_CONFIG_SASL_USERNAME] = self.sasl_username
            if self.sasl_password:
                config[KAFKA_CONFIG_SASL_PASSWORD] = self.sasl_password

        # Apply service-specific overrides
        config.update(overrides)

        return config

    def get_consumer_config(self, group_id: str, **overrides: Any) -> Dict[str, Any]:
        """Build confluent-kafka consumer configuration dictionary.

        Args:
            group_id: Kafka consumer group ID for this consumer.
            **overrides: Additional or override configuration parameters.

        Returns:
            Configuration dict compatible with confluent-kafka Consumer.

        Example:
            ```python
            kafka_config = KafkaConnectionConfig(bootstrap_servers="kafka:9092")
            consumer_config = kafka_config.get_consumer_config(
                group_id="my-service-group",
                max_poll_records=100
            )
            consumer = Consumer(consumer_config)
            ```
        """
        config = {
            KAFKA_CONFIG_BOOTSTRAP_SERVERS: self.bootstrap_servers,
            KAFKA_CONFIG_GROUP_ID: group_id,
            KAFKA_CONFIG_SECURITY_PROTOCOL: self.security_protocol,
            KAFKA_CONFIG_ENABLE_AUTO_COMMIT: self.consumer_enable_auto_commit,
            KAFKA_CONFIG_AUTO_OFFSET_RESET: self.consumer_auto_offset_reset,
            KAFKA_CONFIG_SESSION_TIMEOUT_MS: self.consumer_session_timeout_ms,
            KAFKA_CONFIG_MAX_POLL_INTERVAL_MS: self.consumer_max_poll_interval_ms,
        }

        # Add SASL authentication if required
        if self.security_protocol in [SECURITY_PROTOCOL_SASL_PLAINTEXT, SECURITY_PROTOCOL_SASL_SSL]:
            if self.sasl_mechanism:
                config[KAFKA_CONFIG_SASL_MECHANISM] = self.sasl_mechanism
            if self.sasl_username:
                config[KAFKA_CONFIG_SASL_USERNAME] = self.sasl_username
            if self.sasl_password:
                config[KAFKA_CONFIG_SASL_PASSWORD] = self.sasl_password

        # Apply service-specific overrides
        config.update(overrides)

        return config

class ConsumerFailurePolicy(BaseSchema):
    """Configuration for consumer-side failure handling and retry behavior.

    This policy defines how the consumer should handle message processing failures,
    including retry topic publishing and dead letter queue (DLQ) handling.

    Design Philosophy:
    - Failed messages published to retry topic (observable, durable, non-blocking)
    - Exponential backoff implemented via retry topic consumption delay
    - DLQ for messages exceeding max_retries
    - Config-driven per-topic failure handling

    Retry Flow:
    1. Message fails → publish to retry_topic with metadata headers → commit offset
    2. Retry topic consumer reads message → checks retry count in headers
    3. If retry_attempt <= max_retries: process again
    4. If retry_attempt > max_retries: publish to dlq_topic

    Attributes:
        max_retries: Maximum number of retry attempts before sending to DLQ.
            0 = no retries, immediate DLQ
            N = retry N times via retry_topic before DLQ
        retry_topic: Retry topic name for failed messages (optional).
            If None, no retry mechanism (straight to DLQ or lost).
        dlq_topic: Dead letter queue topic name (optional).
            If None and max_retries exceeded, message is lost.
        track_retry_in_headers: Whether to add retry metadata to message headers.
            Enables stateless retry tracking across consumer restarts.
        retry_backoff_multiplier: Exponential backoff multiplier (e.g., 2.0 = double each retry).
        retry_backoff_base_seconds: Base delay in seconds for first retry.
    """
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retry attempts before DLQ (0 = no retries)"
    )
    retry_topic: Optional[str] = Field(
        default=None,
        description="Retry topic name for failed messages (None = no retry)"
    )
    dlq_topic: Optional[str] = Field(
        default=None,
        description="Dead letter queue topic name (None = no DLQ)"
    )
    track_retry_in_headers: bool = Field(
        default=True,
        description="Add retry count/timestamp to message headers"
    )
    retry_backoff_multiplier: float = Field(
        default=2.0,
        gt=0.0,
        description="Exponential backoff multiplier for retry delays"
    )
    retry_backoff_base_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Base delay in seconds for first retry attempt"
    )


class TopicSubscription(BaseSchema):
    """Configuration for a single Kafka topic subscription.

    Maps a Kafka topic to a handler identifier that will be looked up
    in the DI container's handler registry.

    Attributes:
        topic: Name of the Kafka topic to subscribe to.
        handler_id: Identifier for the handler function in the DI registry.
            This allows topic names to be configured in YAML while handler
            implementations remain in code.
        failure_policy_key: Optional key to lookup failure policy from
            infrastructure.resilience.consumer_failure_policies.
            If None, uses default behavior (infinite retries, no DLQ).
    """
    topic: str = Field(..., description="Kafka topic name")
    handler_id: str = Field(..., description="Handler identifier for DI lookup")
    failure_policy_key: Optional[str] = Field(
        default=None,
        description="Failure policy key from resilience config (optional)"
    )


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
        ...,
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
