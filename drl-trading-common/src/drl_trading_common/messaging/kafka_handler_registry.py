"""Kafka handler registry wrapper for dependency injection compatibility."""

from typing import Dict, Optional

from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler


class KafkaHandlerRegistry:
    """
    Registry for Kafka message handlers.

    Wraps a dictionary of handlers to make it compatible with Python Injector
    which requires @multiprovider for Dict return types.
    """

    def __init__(self, handlers: Dict[str, KafkaMessageHandler]) -> None:
        """
        Initialize the registry.

        Args:
            handlers: Dictionary mapping handler IDs to handler instances
        """
        self._handlers = handlers

    def get_handler(self, handler_id: str) -> Optional[KafkaMessageHandler]:
        """
        Get a handler by ID.

        Args:
            handler_id: The handler identifier

        Returns:
            The handler instance if found, None otherwise
        """
        return self._handlers.get(handler_id)

    def get_all_handlers(self) -> Dict[str, KafkaMessageHandler]:
        """
        Get all registered handlers.

        Returns:
            Dictionary of all handlers
        """
        return self._handlers.copy()

    def __contains__(self, handler_id: str) -> bool:
        """
        Check if a handler is registered.

        Args:
            handler_id: The handler identifier

        Returns:
            True if the handler is registered
        """
        return handler_id in self._handlers

    def __len__(self) -> int:
        """
        Get the number of registered handlers.

        Returns:
            Number of handlers in the registry
        """
        return len(self._handlers)
