"""DRL Trading Common Library.

Shared components for the DRL Trading System including messaging infrastructure,
data models, and utilities.
"""

__version__ = "0.1.0"

# Export main messaging components for easy imports
from .messaging import (
    DeploymentMode,
    Message,
    TradingMessageBus,
    TradingMessageBusFactory,
    TransportInterface,
)

__all__ = [
    # Messaging
    "DeploymentMode",
    "TradingMessageBus",
    "TradingMessageBusFactory",
    "TransportInterface",
    "Message",
]
