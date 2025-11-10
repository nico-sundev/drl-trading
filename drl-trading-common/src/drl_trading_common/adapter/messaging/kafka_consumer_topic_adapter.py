"""Kafka consumer adapter with topic-based handler routing.

This adapter provides configuration-driven message routing based on Kafka topics.
Each topic is mapped to a specific handler function via configuration.
"""

import logging
import threading
from typing import Any, Dict, Optional

from confluent_kafka import Consumer, KafkaError, Message

from drl_trading_common.config.kafka_config import ConsumerFailurePolicy
from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler
from drl_trading_common.messaging.kafka_constants import DEFAULT_POLL_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)


class KafkaConsumerTopicAdapter:
    """Kafka consumer with topic-based handler routing.

    Routes messages to handlers based on Kafka topic configuration.
    Each topic is mapped to exactly one handler function.

    Design: Configuration-driven topic routing
    - Topic subscriptions are defined in YAML config
    - Each topic maps to a handler_id (string key)
    - Handler registry (from DI) maps handler_id to actual handler function
    - This adapter receives the final topic->handler mapping

    Example:
        ```python
        # Build topic->handler mapping from config
        topic_handlers = {
            "requested.preprocess-data": preprocessing_handler,
            "requested.validation": validation_handler,
        }

        # Consumer config
        consumer_config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "my-service",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        # Create and start adapter
        adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=list(topic_handlers.keys()),
            topic_handlers=topic_handlers,
        )
        adapter.start()  # Blocks until stop()
        ```

    Thread Safety:
        Not thread-safe. Run in dedicated thread.
    """

    def __init__(
        self,
        consumer_config: Dict[str, Any],
        topics: list[str],
        topic_handlers: Dict[str, KafkaMessageHandler],
        poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
        failure_policies: Optional[Dict[str, ConsumerFailurePolicy]] = None,
        dlq_producer: Optional[Any] = None,
        retry_producer: Optional[Any] = None,
    ) -> None:
        """Initialize Kafka consumer with topic-based routing, retry topic and DLQ support.

        Args:
            consumer_config: Configuration dict for confluent-kafka Consumer.
                Must include: bootstrap.servers, group.id
            topics: List of topics to subscribe to.
            topic_handlers: Direct topic->handler mapping for routing.
            poll_timeout_seconds: Timeout for poll() calls in seconds.
            failure_policies: Optional dict of topic->failure policy mappings.
                Defines retry limits, retry topic, and DLQ behavior per topic.
            dlq_producer: Optional KafkaProducerAdapter for publishing to DLQ.
                Must be provided if any failure policy specifies a dlq_topic.
            retry_producer: Optional KafkaProducerAdapter for publishing to retry topic.
                Must be provided if any failure policy specifies a retry_topic.

        Raises:
            ValueError: If config or topic_handlers are invalid.
        """
        if not consumer_config:
            raise ValueError("consumer_config cannot be empty")

        if not topics:
            raise ValueError("topics cannot be empty")

        if not topic_handlers:
            raise ValueError("topic_handlers cannot be empty")

        self._consumer: Optional[Consumer] = Consumer(consumer_config)
        self._topics = topics
        self._topic_handlers = topic_handlers
        self._poll_timeout = poll_timeout_seconds
        self._running = False
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()

        # Retry and DLQ infrastructure
        self._failure_policies = failure_policies or {}
        self._dlq_producer = dlq_producer
        self._retry_producer = retry_producer

        # Extract group ID for logging
        self._group_id = consumer_config.get("group.id", "unknown")

        logger.info(
            "KafkaConsumerTopicAdapter initialized",
            extra={
                "group_id": self._group_id,
                "topics": self._topics,
                "handler_count": len(topic_handlers),
                "poll_timeout_seconds": poll_timeout_seconds,
                "failure_policies_count": len(self._failure_policies),
                "retry_enabled": retry_producer is not None,
                "dlq_enabled": dlq_producer is not None,
            }
        )

    def start(self) -> None:
        """Start consuming messages from subscribed topics.

        This method blocks until stop() is called or a fatal error occurs.
        Run in a separate thread if you need non-blocking behavior.

        Raises:
            RuntimeError: If consumer is already running.
            Exception: If a fatal error occurs in the consumer loop.
        """
        if self._running:
            raise RuntimeError("Consumer is already running")

        if not self._consumer:
            raise RuntimeError("Consumer has been closed")

        self._consumer.subscribe(self._topics)
        self._running = True

        logger.info(
            "Kafka consumer started and subscribed to topics",
            extra={
                "group_id": self._group_id,
                "topics": self._topics,
                "handler_count": len(self._topic_handlers),
            }
        )

        try:
            while self._running:
                # If a shutdown was requested via signal handler, stop gracefully
                if self._shutdown_event.is_set():
                    # Call stop() from this thread (not the signal handler)
                    self.stop()
                    break
                msg = self._consumer.poll(timeout=self._poll_timeout)

                if msg is None:
                    # No message within timeout - continue polling
                    continue

                if msg.error():
                    self._handle_kafka_error(msg)
                    continue

                # Route message to handler based on header
                self._process_message(msg)

        except Exception as e:
            logger.error(
                "Fatal error in Kafka consumer loop",
                extra={"group_id": self._group_id, "error": str(e)},
                exc_info=True
            )
            raise

        finally:
            self._cleanup()

    def _handle_kafka_error(self, msg: Message) -> None:
        """Handle Kafka-specific errors.

        Args:
            msg: Message object containing error information.
        """
        error = msg.error()

        if error.code() == KafkaError._PARTITION_EOF:
            # Reached end of partition - not an error, just informational
            logger.debug(
                "Reached end of partition",
                extra={
                    "group_id": self._group_id,
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                }
            )
        else:
            logger.error(
                "Kafka consumer error",
                extra={
                    "group_id": self._group_id,
                    "error_code": error.code(),
                    "error_name": error.name(),
                    "error_message": error.str(),
                    "topic": msg.topic() if msg.topic() else "unknown",
                }
            )

    def _process_message(self, msg: Message) -> None:
        """Process a single message by routing to handler based on topic.

        Routing logic:
        1. Look up handler for message topic
        2. Invoke handler
        3. Commit offset on success
        4. Send to DLQ on failure (TODO: implement DLQ)

        Args:
            msg: The Kafka message to process.
        """
        topic = msg.topic()

        # Look up handler for this topic
        handler = self._topic_handlers.get(topic)
        if not handler:
            self._skip_message_no_handler(msg, topic)
            return

        # Execute handler with error handling
        try:
            self._execute_handler(handler, msg, topic)
        except Exception as e:
            self._handle_processing_error(msg, topic, e)

    def _skip_message_no_handler(self, msg: Message, topic: str) -> None:
        """Skip message when no handler is configured for the topic.

        Args:
            msg: The Kafka message to skip.
            topic: The topic name.
        """
        logger.warning(
            "No handler configured for topic - skipping",
            extra={
                "group_id": self._group_id,
                "topic": topic,
                "partition": msg.partition(),
                "offset": msg.offset(),
                "configured_topics": list(self._topic_handlers.keys()),
            }
        )
        self._commit_offset(msg)

    def _execute_handler(self, handler: KafkaMessageHandler, msg: Message, topic: str) -> None:
        """Execute the handler and commit offset on success.

        Args:
            handler: The handler function to execute.
            msg: The Kafka message to process.
            topic: The topic name.

        Raises:
            Exception: Re-raises any exception from handler execution.
        """
        handler(msg)
        self._commit_offset(msg)

        logger.debug(
            "Message processed successfully",
            extra={
                "group_id": self._group_id,
                "topic": topic,
                "partition": msg.partition(),
                "offset": msg.offset(),
                "key": msg.key().decode("utf-8") if msg.key() else None,
            }
        )

    def _handle_processing_error(self, msg: Message, topic: str, error: Exception) -> None:
        """Handle errors during message processing with retry topic and DLQ.

        Strategy:
        1. Extract retry_attempt from message headers (stateless tracking)
        2. If retry_attempt < max_retries AND retry_topic configured:
           → Publish to retry_topic with incremented retry_attempt header
           → Commit offset (non-blocking, durable retry)
        3. If retry_attempt >= max_retries:
           → Publish to DLQ if configured
           → Commit offset
        4. If no retry/DLQ configured:
           → Log error and commit (message lost)

        Args:
            msg: The Kafka message that failed processing.
            topic: The topic name.
            error: The exception that occurred.
        """
        from drl_trading_common.adapter.messaging.retry_metadata import (
            extract_retry_attempt,
            extract_header_value,
            HEADER_FIRST_FAILURE_TIMESTAMP,
        )

        policy = self._failure_policies.get(topic)

        if not policy:
            # No policy configured = log and commit (fail-fast, don't block queue)
            logger.error(
                "Handler failed, no failure policy configured - committing offset (message lost)",
                extra={
                    "group_id": self._group_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                exc_info=True,
            )
            self._commit_offset(msg)
            return

        # Extract retry attempt from headers (0 if first failure)
        retry_attempt = extract_retry_attempt(msg.headers())
        new_retry_attempt = retry_attempt + 1  # This failure counts as an attempt
        first_failure_timestamp = extract_header_value(msg.headers(), HEADER_FIRST_FAILURE_TIMESTAMP)

        if retry_attempt < policy.max_retries and policy.retry_topic:
            # Publish to retry topic with incremented attempt
            logger.warning(
                f"Handler failed (attempt {new_retry_attempt}/{policy.max_retries}) - publishing to retry topic",
                extra={
                    "group_id": self._group_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_attempt": new_retry_attempt,
                    "max_retries": policy.max_retries,
                    "retry_topic": policy.retry_topic,
                },
                exc_info=True,
            )

            self._publish_to_retry_topic(
                msg=msg,
                original_topic=topic,
                error=error,
                retry_topic=policy.retry_topic,
                retry_attempt=new_retry_attempt,
                first_failure_timestamp=first_failure_timestamp,
                backoff_base=policy.retry_backoff_base_seconds,
                backoff_multiplier=policy.retry_backoff_multiplier,
            )
            self._commit_offset(msg)
            return

        # Max retries exceeded or no retry topic configured
        logger.error(
            f"Max retries ({policy.max_retries}) exceeded or no retry topic configured",
            extra={
                "group_id": self._group_id,
                "topic": topic,
                "partition": msg.partition(),
                "offset": msg.offset(),
                "retry_attempts": retry_attempt,
                "has_retry_topic": policy.retry_topic is not None,
            },
        )

        if policy.dlq_topic and self._dlq_producer:
            # Publish to DLQ with incremented retry count (this failure counts as an attempt)
            self._publish_to_dlq(msg, topic, error, policy.dlq_topic, new_retry_attempt)
        elif policy.dlq_topic and not self._dlq_producer:
            logger.critical(
                "DLQ topic configured but no DLQ producer available - message will be lost!",
                extra={
                    "topic": topic,
                    "offset": msg.offset(),
                    "dlq_topic": policy.dlq_topic,
                },
            )
        else:
            logger.warning(
                "No DLQ configured - committing offset, message will be lost",
                extra={"topic": topic, "offset": msg.offset()},
            )

        self._commit_offset(msg)

    def _publish_to_retry_topic(
        self,
        msg: Message,
        original_topic: str,
        error: Exception,
        retry_topic: str,
        retry_attempt: int,
        first_failure_timestamp: str | None,
        backoff_base: float,
        backoff_multiplier: float,
    ) -> None:
        """Publish failed message to retry topic with metadata headers.

        The message is published to the retry topic with headers containing:
        - Retry attempt number
        - Original topic
        - Error information
        - Timestamps for tracking
        - Next retry timestamp (for exponential backoff)

        Args:
            msg: The original Kafka message that failed.
            original_topic: The topic where the message came from.
            error: The exception that caused the failure.
            retry_topic: The retry topic name.
            retry_attempt: Current retry attempt number (1-indexed).
            first_failure_timestamp: ISO timestamp of first failure (None for first attempt).
            backoff_base: Base delay in seconds for exponential backoff.
            backoff_multiplier: Multiplier for exponential backoff.
        """
        from drl_trading_common.adapter.messaging.retry_metadata import (
            build_retry_headers,
            calculate_backoff_seconds,
        )

        if not self._retry_producer:
            logger.critical(
                "Cannot publish to retry topic - no retry producer available",
                extra={"topic": original_topic, "offset": msg.offset()},
            )
            return

        try:
            # Calculate exponential backoff delay
            backoff_delay = calculate_backoff_seconds(
                retry_attempt=retry_attempt,
                base_seconds=backoff_base,
                multiplier=backoff_multiplier,
            )

            # Build retry metadata headers
            retry_headers_list = build_retry_headers(
                retry_attempt=retry_attempt,
                original_topic=original_topic,
                error_type=type(error).__name__,
                error_message=str(error),
                first_failure_timestamp=first_failure_timestamp,
                next_retry_after_seconds=backoff_delay,
            )

            # Convert list of tuples to dict for producer (decode bytes values)
            retry_headers_dict = {key: value.decode("utf-8") for key, value in retry_headers_list}

            # Decode original message value from JSON to dict for republishing
            import json
            try:
                original_value_dict = json.loads(msg.value().decode("utf-8")) if msg.value() else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
                logger.error(
                    f"Failed to decode message value for retry topic - message may be corrupted: {decode_error}",
                    extra={"original_topic": original_topic, "offset": msg.offset()},
                )
                # If we can't decode, wrap the raw bytes as base64 string to preserve data
                import base64
                original_value_dict = {
                    "_raw_value_base64": base64.b64encode(msg.value()).decode("utf-8") if msg.value() else "",
                    "_decode_error": str(decode_error),
                }

            # Publish original message value to retry topic with metadata headers
            self._retry_producer.publish(
                topic=retry_topic,
                key=msg.key().decode("utf-8") if msg.key() else None,
                value=original_value_dict,  # Pass as dict, producer will re-serialize
                headers=retry_headers_dict,
            )

            logger.info(
                f"Published message to retry topic (attempt {retry_attempt}, backoff {backoff_delay:.1f}s)",
                extra={
                    "original_topic": original_topic,
                    "retry_topic": retry_topic,
                    "retry_attempt": retry_attempt,
                    "backoff_seconds": backoff_delay,
                    "offset": msg.offset(),
                },
            )
        except Exception as publish_error:
            logger.error(
                "Failed to publish to retry topic",
                extra={
                    "original_topic": original_topic,
                    "retry_topic": retry_topic,
                    "error": str(publish_error),
                },
                exc_info=True,
            )

    def _publish_to_dlq(
        self,
        msg: Message,
        original_topic: str,
        error: Exception,
        dlq_topic: str,
        retry_attempts: int,
    ) -> None:
        """Publish failed message to DLQ with metadata.

        Uses the injected KafkaProducerAdapter which has tenacity retry logic built-in.

        Args:
            msg: The original Kafka message that failed.
            original_topic: The topic where the message came from.
            error: The exception that caused the failure.
            dlq_topic: The dead letter queue topic name.
            retry_attempts: Number of retry attempts made.
        """
        if not self._dlq_producer:
            logger.critical(
                "Cannot publish to DLQ - no producer available",
                extra={"topic": original_topic, "offset": msg.offset()},
            )
            return

        try:
            # Decode original message value (assuming UTF-8 JSON)
            original_value = msg.value().decode("utf-8") if msg.value() else ""

            # Build DLQ payload with metadata
            dlq_payload = {
                "original_topic": original_topic,
                "original_partition": msg.partition(),
                "original_offset": msg.offset(),
                "original_timestamp": msg.timestamp()[1] if msg.timestamp()[0] != -1 else None,
                "original_key": msg.key().decode("utf-8") if msg.key() else None,
                "original_value": original_value,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "retry_attempts": retry_attempts,
                "consumer_group": self._group_id,
            }

            # Publish using DLQ producer (which has tenacity retry built-in)
            self._dlq_producer.publish(
                topic=dlq_topic,
                key=msg.key().decode("utf-8") if msg.key() else "unknown",
                value=dlq_payload,
                headers={
                    "original_topic": original_topic,
                    "error_type": type(error).__name__,
                    "retry_attempts": str(retry_attempts),
                },
            )

            logger.info(
                f"Successfully published message to DLQ: {dlq_topic}",
                extra={
                    "original_topic": original_topic,
                    "original_offset": msg.offset(),
                    "dlq_topic": dlq_topic,
                },
            )

        except Exception as dlq_error:
            logger.critical(
                "CRITICAL: Failed to publish message to DLQ - message will be lost!",
                extra={
                    "original_topic": original_topic,
                    "original_offset": msg.offset(),
                    "dlq_topic": dlq_topic,
                    "dlq_error": str(dlq_error),
                },
                exc_info=True,
            )
            # Note: We still commit the offset below to avoid infinite loop

    def _commit_offset(self, msg: Message) -> None:
        """Commit the message offset.

        Args:
            msg: The Kafka message whose offset should be committed.
        """
        if self._consumer:
            self._consumer.commit(msg)

    def stop(self) -> None:
        """Signal the consumer to stop gracefully.

        Thread-safe: Can be called from other threads.
        Sets both the shutdown event and running flag to ensure the
        consumer loop terminates promptly.
        """
        with self._shutdown_lock:
            if self._running:
                logger.info(
                    "Stopping Kafka consumer...",
                    extra={"group_id": self._group_id}
                )
                self._running = False
                self._shutdown_event.set()  # Signal the consumer loop to exit

    def _cleanup(self) -> None:
        """Clean up consumer resources.

        Called automatically in finally block of start() method.
        """
        logger.info(
            "Cleaning up Kafka consumer resources...",
            extra={"group_id": self._group_id}
        )

        if self._consumer:
            try:
                self._consumer.close()
                logger.info(
                    "Kafka consumer closed successfully",
                    extra={"group_id": self._group_id}
                )
            except Exception as e:
                logger.error(
                    "Error closing Kafka consumer",
                    extra={
                        "group_id": self._group_id,
                        "error": str(e),
                    },
                    exc_info=True
                )
            finally:
                self._consumer = None
