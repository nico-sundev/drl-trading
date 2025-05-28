"""Transport abstraction for different deployment scenarios."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class DeploymentMode(Enum):
    """Deployment mode configuration."""

    TRAINING = "training"  # Direct function calls, single process
    PRODUCTION = "production"  # Message queue, distributed


@dataclass
class Message:
    """Base message structure."""

    topic: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class TransportInterface(ABC):
    """Abstract transport layer for messages."""

    @abstractmethod
    def publish(self, message: Message) -> None:
        """Publish a message."""
        pass

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to a topic with a handler."""
        pass

    @abstractmethod
    def request_reply(self, message: Message, timeout_seconds: int = 30) -> Message:
        """Send request and wait for reply (RPC pattern)."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the transport."""
        pass


class TransportFactory:
    """Factory for creating transport instances based on deployment mode."""

    @staticmethod
    def create_transport(mode: DeploymentMode, **kwargs) -> TransportInterface:
        """Create transport based on deployment mode."""
        if mode == DeploymentMode.TRAINING:
            from .in_memory_transport import InMemoryTransport

            return InMemoryTransport()
        elif mode == DeploymentMode.PRODUCTION:
            from .rabbitmq_transport import RabbitMQTransport

            return RabbitMQTransport(**kwargs)
        else:
            raise ValueError(f"Unknown deployment mode: {mode}")
