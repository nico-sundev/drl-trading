"""Kafka producer adapter with retry and resilience capabilities.

This module provides a generic Kafka producer implementation that integrates
with the tenacity library for retry logic and error handling. It supports
JSON serialization, dead letter queue (DLQ) publishing, and health checking.
"""

import logging
from collections.abc import Callable
from typing import Any, Dict, Optional

from confluent_kafka import KafkaException, Producer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random,
)

from drl_trading_common.config.resilience_config import RetryConfig
from drl_trading_common.messaging.kafka_serializers import serialize_to_json

logger = logging.getLogger(__name__)


class KafkaProducerAdapter:
    """Generic Kafka producer with retry logic and DLQ support.

    This adapter wraps confluent-kafka's Producer with configurable retry
    behavior using tenacity. It provides JSON serialization, error handling,
    and support for dead letter queues.

    The retry configuration is provided at construction time, allowing each
    producer instance to have different resilience behavior based on its
    use case (e.g., critical data vs. best-effort notifications).

    Attributes:
        _producer: The underlying confluent-kafka Producer instance.
        _retry_config: Configuration for retry behavior.
        _dlq_topic: Optional topic name for failed messages.
    """

    def __init__(
        self,
        producer_config: Dict[str, Any],
        retry_config: RetryConfig,
        dlq_topic: Optional[str] = None,
    ) -> None:
        """Initialize the Kafka producer adapter.

        Args:
            producer_config: Configuration dict for confluent-kafka Producer.
                Should be obtained from KafkaConnectionConfig.get_producer_config().
            retry_config: Configuration for retry behavior (attempts, backoff, etc.).
            dlq_topic: Optional topic name for publishing failed messages.
                If provided, messages that fail after all retries will be sent here.
        """
        self._producer = Producer(producer_config)
        self._retry_config = retry_config
        self._dlq_topic = dlq_topic
        logger.info(
            f"KafkaProducerAdapter initialized with max_attempts={retry_config.max_attempts}, "
            f"dlq_topic={dlq_topic or 'none'}"
        )

    def _create_retry_decorator(self) -> Callable[[Callable], Callable]:
        """Create a tenacity retry decorator based on the retry config.

        Returns:
            A configured retry decorator that can be applied to methods.
        """
        stop_strategy: Any = stop_after_attempt(self._retry_config.max_attempts)

        # Add optional stop_after_delay if configured
        if self._retry_config.stop_after_delay is not None:
            stop_strategy = stop_strategy | stop_after_delay(
                self._retry_config.stop_after_delay
            )

        wait_strategy = wait_exponential(
            multiplier=self._retry_config.wait_exponential_multiplier,
            max=self._retry_config.wait_exponential_max,
        ) + wait_random(0, self._retry_config.wait_jitter_max)

        return retry(
            retry=retry_if_exception_type(KafkaException),
            stop=stop_strategy,
            wait=wait_strategy,
            reraise=True,
        )

    def publish(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a message to a Kafka topic with retry logic.

        The message value is automatically serialized to JSON. If publishing
        fails after all retries and a DLQ topic is configured, the message
        will be sent to the DLQ.

        Args:
            topic: The Kafka topic to publish to.
            key: The message key (used for partitioning).
            value: The message payload as a dictionary (will be JSON-serialized).
            headers: Optional message headers as key-value pairs.

        Raises:
            KafkaException: If publishing fails after all retry attempts and
                no DLQ is configured, or if DLQ publishing also fails.
        """
        try:
            self._publish_with_retry(topic, key, value, headers)
        except KafkaException as e:
            logger.error(
                f"Failed to publish message to topic '{topic}' after {self._retry_config.max_attempts} attempts: {e}"
            )
            if self._dlq_topic:
                logger.warning(
                    f"Attempting to publish failed message to DLQ: {self._dlq_topic}"
                )
                self._publish_to_dlq(topic, key, value, headers, str(e))
            else:
                raise

    def _publish_with_retry(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Internal method that applies retry logic to publishing.

        This method is decorated dynamically with retry behavior based on
        the retry configuration.

        Args:
            topic: The Kafka topic to publish to.
            key: The message key.
            value: The message payload dictionary.
            headers: Optional message headers.

        Raises:
            KafkaException: If publishing fails after all retries.
        """
        retry_decorator = self._create_retry_decorator()
        retry_decorated_publish = retry_decorator(self._produce_message)
        retry_decorated_publish(topic, key, value, headers)

    def _produce_message(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Produce a message to Kafka (low-level operation).

        Args:
            topic: The Kafka topic.
            key: The message key.
            value: The message payload dictionary.
            headers: Optional message headers.

        Raises:
            KafkaException: If the producer encounters an error.
        """
        serialized_value = serialize_to_json(value)
        kafka_headers = (
            [(k, v.encode("utf-8")) for k, v in headers.items()] if headers else None
        )

        self._producer.produce(
            topic=topic,
            key=key.encode("utf-8") if key else None,
            value=serialized_value,
            headers=kafka_headers,
            callback=self._delivery_callback,
        )
        self._producer.poll(0)  # Trigger callbacks

    def _delivery_callback(self, err: Optional[Exception], msg: Any) -> None:
        """Callback for Kafka producer delivery reports.

        Args:
            err: Error if delivery failed, None if successful.
            msg: The message that was delivered or failed.
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
            raise KafkaException(err)
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} [partition {msg.partition()}] at offset {msg.offset()}"
            )

    def _publish_to_dlq(
        self,
        original_topic: str,
        key: str,
        value: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        error_message: str = "",
    ) -> None:
        """Publish a failed message to the dead letter queue.

        Args:
            original_topic: The original topic where publishing failed.
            key: The message key.
            value: The message payload.
            headers: Optional original headers.
            error_message: Description of the error that caused the failure.
        """
        if not self._dlq_topic:
            return

        dlq_headers = headers.copy() if headers else {}
        dlq_headers.update(
            {
                "original_topic": original_topic,
                "error_message": error_message,
            }
        )

        try:
            # DLQ publishing should use minimal retry (typically configured with max_attempts=1)
            self._produce_message(self._dlq_topic, key, value, dlq_headers)
            self.flush()
            logger.info(f"Message successfully published to DLQ: {self._dlq_topic}")
        except KafkaException as dlq_error:
            logger.critical(
                f"Failed to publish message to DLQ '{self._dlq_topic}': {dlq_error}. "
                f"Message lost: topic={original_topic}, key={key}"
            )
            raise

    def flush(self, timeout: float = 10.0) -> int:
        """Flush all pending messages to Kafka.

        This method blocks until all buffered messages are sent or the timeout expires.

        Args:
            timeout: Maximum time to wait for flush completion (seconds).

        Returns:
            Number of messages still in queue after timeout.
        """
        remaining: int = self._producer.flush(timeout)
        if remaining > 0:
            logger.warning(
                f"{remaining} messages remaining in queue after flush timeout"
            )
        return remaining

    def health_check(self) -> bool:
        """Check if the Kafka producer is healthy and can connect to the cluster.

        Returns:
            True if the producer is healthy, False otherwise.
        """
        try:
            # Attempt to get cluster metadata as a health check
            metadata = self._producer.list_topics(timeout=5.0)
            logger.debug(f"Kafka health check successful. Cluster has {len(metadata.topics)} topics")
            return True
        except KafkaException as e:
            logger.error(f"Kafka health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the Kafka producer and flush all pending messages.

        This should be called during application shutdown to ensure all
        messages are delivered before terminating.
        """
        logger.info("Closing KafkaProducerAdapter...")
        remaining = self.flush(timeout=30.0)
        if remaining > 0:
            logger.warning(
                f"KafkaProducerAdapter closed with {remaining} messages still in queue"
            )
        else:
            logger.info("KafkaProducerAdapter closed successfully")
