"""Constants for retry configuration keys in the preprocess service.

This module defines service-specific constants that map use cases to
retry configuration keys. These constants are referenced in the service
configuration and dependency injection modules to ensure type-safe
access to retry policies.
"""

# Retry config key for publishing resampled market data to Kafka
RETRY_CONFIG_KAFKA_RESAMPLED_DATA = "kafka_resampled_data_retry"

# Retry config key for publishing preprocessing completion events to Kafka
RETRY_CONFIG_KAFKA_PREPROCESSING_COMPLETED = "kafka_preprocessing_completed_retry"

# Retry config key for publishing to dead letter queue (minimal retries)
RETRY_CONFIG_KAFKA_DLQ = "kafka_dlq_retry"
