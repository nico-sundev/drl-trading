"""Kafka consumer adapter with header-based handler routing.

This adapter extends the base consumer pattern with header-based routing,
enabling multiple handlers per topic based on 'handler_id' message header.
"""

import logging
import threading
from typing import Any, Dict, Optional, Union

from confluent_kafka import Consumer, KafkaError, Message

from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler
from drl_trading_common.messaging.kafka_handler_registry import KafkaHandlerRegistry
from drl_trading_common.messaging.kafka_constants import DEFAULT_POLL_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)

HEADER_HANDLER_ID = "handler_id"


class KafkaConsumerHeaderAdapter:
    """Kafka consumer with header-based handler routing.

    Unlike the topic-based adapter, this adapter:
    - Routes messages by 'handler_id' header instead of topic
    - Supports multiple handlers per topic
    - Enables configuration-driven handler selection

    Design: Configuration-driven routing
    - Producers specify handler_id in message headers
    - Consumer looks up handler from registry using header value
    - Decouples topic names from handler implementations

    Example:
        ```python
        # Create handler registry
        handler_registry = {
            "preprocessing-request": preprocessing_handler,
            "validation-request": validation_handler,
        }

        # Consumer config
        consumer_config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "my-service",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        # Create and start adapter
        adapter = KafkaConsumerHeaderAdapter(
            consumer_config=consumer_config,
            topics=["requests"],
            handler_registry=handler_registry,
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
        handler_registry: Union[Dict[str, KafkaMessageHandler], KafkaHandlerRegistry],
        poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize Kafka consumer with header-based routing.

        Args:
            consumer_config: Configuration dict for confluent-kafka Consumer.
                Must include: bootstrap.servers, group.id
            topics: List of topics to subscribe to.
            handler_registry: Either a dict mapping handler_id to handler,
                or a KafkaHandlerRegistry instance.
            poll_timeout_seconds: Timeout for poll() calls in seconds.

        Raises:
            ValueError: If config or handler_registry are invalid.
        """
        if not consumer_config:
            raise ValueError("consumer_config cannot be empty")

        if not topics:
            raise ValueError("topics cannot be empty")

        if not handler_registry:
            raise ValueError("handler_registry cannot be empty")

        # Normalize registry to dict for internal use
        if isinstance(handler_registry, KafkaHandlerRegistry):
            handlers_dict = handler_registry.get_all_handlers()
        else:
            handlers_dict = handler_registry

        self._consumer: Optional[Consumer] = Consumer(consumer_config)
        self._topics = topics
        self._handler_registry = handlers_dict
        self._poll_timeout = poll_timeout_seconds
        self._running = False
        # Event set by signal handler to request shutdown (safe to set from signal handler)
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()

        # Extract group ID for logging
        self._group_id = consumer_config.get("group.id", "unknown")

        # Note: Signal handler registration removed. The consumer should be stopped
        # by the service's shutdown lifecycle (_stop_service) rather than registering
        # its own handlers. Multiple signal handlers can conflict and prevent proper
        # service shutdown coordination.

        logger.info(
            "KafkaConsumerHeaderAdapter initialized",
            extra={
                "group_id": self._group_id,
                "topics": self._topics,
                "handler_ids": list(handlers_dict.keys()),
                "poll_timeout_seconds": poll_timeout_seconds,
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
                "handler_count": len(self._handler_registry),
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
        """Process a single message by routing to handler via header.

        Routing logic:
        1. Extract 'handler_id' from message headers
        2. Look up handler in registry
        3. Invoke handler
        4. Commit offset on success

        Args:
            msg: The Kafka message to process.
        """
        topic = msg.topic()
        headers = msg.headers() or []

        # Extract handler_id from headers
        handler_id = None
        for key, value in headers:
            if key == HEADER_HANDLER_ID:
                handler_id = value.decode("utf-8") if isinstance(value, bytes) else value
                break

        if not handler_id:
            logger.warning(
                "Message missing 'handler_id' header - skipping",
                extra={
                    "group_id": self._group_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "headers": [(k, v.decode("utf-8") if isinstance(v, bytes) else v) for k, v in headers],
                }
            )
            # Commit offset to avoid reprocessing invalid message
            if self._consumer:
                self._consumer.commit(msg)
            return

        handler = self._handler_registry.get(handler_id)

        if not handler:
            logger.warning(
                "No handler registered for handler_id - skipping",
                extra={
                    "group_id": self._group_id,
                    "handler_id": handler_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "registered_handlers": list(self._handler_registry.keys()),
                }
            )
            # Commit offset to avoid reprocessing unhandled message
            if self._consumer:
                self._consumer.commit(msg)
            return

        try:
            # Invoke handler (business logic)
            handler(msg)

            # Manual commit after successful processing
            if self._consumer:
                self._consumer.commit(msg)

            logger.debug(
                "Message processed and committed successfully",
                extra={
                    "group_id": self._group_id,
                    "handler_id": handler_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "key": msg.key().decode("utf-8") if msg.key() else None,
                }
            )

        except Exception as e:
            logger.error(
                "Handler failed to process message - offset NOT committed",
                extra={
                    "group_id": self._group_id,
                    "handler_id": handler_id,
                    "topic": topic,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True
            )
            # Don't commit offset - message will be reprocessed
            # TODO: Implement DLQ (Dead Letter Queue) after N retries

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
