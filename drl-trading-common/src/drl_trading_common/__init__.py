"""DRL Trading Common Library.

Shared components for the DRL Trading System including messaging infrastructure,
data models, and utilities.
"""

__version__ = "0.1.0"

# Export main messaging components for easy imports
from .messaging import (
    Message,
    TradingMessageBus,
    TradingMessageBusFactory,
    TransportInterface,
)
from .base import BaseFeature, BaseParameterSetConfig, BaseTradingEnv

__all__ = [
    # Messaging
    "TradingMessageBus",
    "TradingMessageBusFactory",
    "TransportInterface",
    "Message",
    # Base components
    "BaseFeature",
    "BaseParameterSetConfig",
    "BaseTradingEnv",
]
