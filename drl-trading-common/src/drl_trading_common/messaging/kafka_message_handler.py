"""Protocol definition for Kafka message handlers.

This module defines the typing protocol that all Kafka message handlers
must conform to, enabling type-safe handler registration and invocation.
"""

from typing import Protocol
from confluent_kafka import Message


class KafkaMessageHandler(Protocol):
    """Protocol for Kafka message handler functions.

    This is a typing.Protocol, not an abstract base class. Handlers don't
    need to explicitly inherit from this - any callable with a matching
    signature automatically satisfies this protocol.

    Handlers are responsible for:
    1. Deserializing the Kafka message payload
    2. Delegating to business logic in the core layer
    3. Raising exceptions on failure (adapter handles retries/DLQ)

    Handlers should NOT:
    1. Commit offsets (adapter handles this)
    2. Contain business logic (delegate to core services)
    3. Handle Kafka-specific errors (adapter's responsibility)

    Example:
        ```python
        def my_handler(message: Message) -> None:
            # Parse message
            data = json.loads(message.value().decode('utf-8'))

            # Delegate to business logic
            my_service.process(data)

            # Exceptions propagate to adapter for retry/DLQ handling
        ```

    Type Safety:
        ```python
        handler: KafkaMessageHandler = my_handler  # Type-checked by mypy
        handler_registry: Dict[str, KafkaMessageHandler] = {
            "handler_id": my_handler
        }
        ```
    """

    def __call__(self, message: Message) -> None:
        """Process a Kafka message.

        Args:
            message: The Kafka message to process. Contains:
                - value(): Message payload (bytes)
                - key(): Message key (bytes or None)
                - topic(): Topic name
                - partition(): Partition number
                - offset(): Message offset
                - timestamp(): Message timestamp
                - headers(): Message headers

        Raises:
            Exception: Any exception raised will be caught by the adapter.
                The adapter will log the error and decide on retry/DLQ behavior.
                The offset will NOT be committed, so the message will be reprocessed.

        Returns:
            None: Handlers should not return values. Side effects (DB writes,
                service calls, etc.) are the expected outcome.
        """
        ...
