"""Kafka configuration constants and default values.

This module contains all string constants used in Kafka configuration
to avoid hardcoding values throughout the codebase.
"""

# Kafka configuration keys (as used by confluent-kafka library)
KAFKA_CONFIG_BOOTSTRAP_SERVERS = "bootstrap.servers"
KAFKA_CONFIG_GROUP_ID = "group.id"
KAFKA_CONFIG_SECURITY_PROTOCOL = "security.protocol"
KAFKA_CONFIG_SASL_MECHANISM = "sasl.mechanism"
KAFKA_CONFIG_SASL_USERNAME = "sasl.username"
KAFKA_CONFIG_SASL_PASSWORD = "sasl.password"

# Producer-specific config keys
KAFKA_CONFIG_ACKS = "acks"
KAFKA_CONFIG_RETRIES = "retries"
KAFKA_CONFIG_COMPRESSION_TYPE = "compression.type"
KAFKA_CONFIG_MAX_IN_FLIGHT = "max.in.flight.requests.per.connection"
KAFKA_CONFIG_LINGER_MS = "linger.ms"
KAFKA_CONFIG_BATCH_SIZE = "batch.size"

# Consumer-specific config keys
KAFKA_CONFIG_ENABLE_AUTO_COMMIT = "enable.auto.commit"
KAFKA_CONFIG_AUTO_OFFSET_RESET = "auto.offset.reset"
KAFKA_CONFIG_SESSION_TIMEOUT_MS = "session.timeout.ms"
KAFKA_CONFIG_MAX_POLL_INTERVAL_MS = "max.poll.interval.ms"
KAFKA_CONFIG_MAX_POLL_RECORDS = "max.poll.records"

# Security protocol values
SECURITY_PROTOCOL_PLAINTEXT = "PLAINTEXT"
SECURITY_PROTOCOL_SSL = "SSL"
SECURITY_PROTOCOL_SASL_PLAINTEXT = "SASL_PLAINTEXT"
SECURITY_PROTOCOL_SASL_SSL = "SASL_SSL"

# SASL mechanism values
SASL_MECHANISM_PLAIN = "PLAIN"
SASL_MECHANISM_SCRAM_SHA_256 = "SCRAM-SHA-256"
SASL_MECHANISM_SCRAM_SHA_512 = "SCRAM-SHA-512"
SASL_MECHANISM_GSSAPI = "GSSAPI"

# Compression types
COMPRESSION_TYPE_NONE = "none"
COMPRESSION_TYPE_GZIP = "gzip"
COMPRESSION_TYPE_SNAPPY = "snappy"
COMPRESSION_TYPE_LZ4 = "lz4"
COMPRESSION_TYPE_ZSTD = "zstd"

# Offset reset strategies
OFFSET_RESET_EARLIEST = "earliest"
OFFSET_RESET_LATEST = "latest"
OFFSET_RESET_NONE = "none"

# Acks values
ACKS_ALL = "all"
ACKS_NONE = "0"
ACKS_LEADER = "1"

# Default configuration values
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_SECURITY_PROTOCOL = SECURITY_PROTOCOL_PLAINTEXT
DEFAULT_CONSUMER_GROUP_ID = "drl-trading-consumer-group"
DEFAULT_CONSUMER_AUTO_OFFSET_RESET = OFFSET_RESET_LATEST
DEFAULT_CONSUMER_ENABLE_AUTO_COMMIT = False  # Manual commit for reliability
DEFAULT_CONSUMER_SESSION_TIMEOUT_MS = 30000  # 30 seconds
DEFAULT_CONSUMER_MAX_POLL_INTERVAL_MS = 300000  # 5 minutes
DEFAULT_PRODUCER_ACKS = ACKS_ALL
DEFAULT_PRODUCER_RETRIES = 3
DEFAULT_PRODUCER_COMPRESSION_TYPE = COMPRESSION_TYPE_SNAPPY
DEFAULT_POLL_TIMEOUT_SECONDS = 1.0
