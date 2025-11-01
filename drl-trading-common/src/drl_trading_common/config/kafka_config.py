
"""Kafka configuration for messaging infrastructure."""
from typing import Optional, Dict, Any

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
