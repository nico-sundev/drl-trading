"""Messaging system for DRL Trading."""

from .in_memory_transport import InMemoryTransport
from .trading_message_bus import TradingMessageBus, TradingMessageBusFactory
from .transport_interface import DeploymentMode, Message, TransportInterface

# Only import RabbitMQ if available
try:
    from .rabbitmq_transport import RabbitMQTransport


except ImportError:
    __all__ = [
        "DeploymentMode",
        "TransportInterface",
        "Message",
        "TradingMessageBus",
        "TradingMessageBusFactory",
        "InMemoryTransport",
        "RabbitMQTransport"
    ]
