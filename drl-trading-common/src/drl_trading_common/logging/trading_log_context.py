"""
Trading-specific logging context management.

Provides thread-local storage for trading context information that flows
through distributed service calls, enabling correlation tracking and
structured logging across the DRL trading microservice ecosystem.
"""

import uuid
from typing import Optional, Dict, Any
from threading import local

from drl_trading_common.adapter.model.trading_context import TradingContext


class TradingLogContext:
    """
    Thread-local storage for trading-specific logging context.

    This class manages context information that should be automatically
    included in all log entries within a thread. It supports progressive
    context enrichment as requests flow through different services.
    """

    _local = local()

    # Core context methods
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for tracking business operations across services."""
        cls._local.correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(cls._local, 'correlation_id', None)

    @classmethod
    def set_event_id(cls, event_id: str) -> None:
        """Set event ID for individual message/event tracking."""
        cls._local.event_id = event_id

    @classmethod
    def get_event_id(cls) -> Optional[str]:
        """Get event ID for current thread."""
        return getattr(cls._local, 'event_id', None)

    # Trading-specific context methods
    @classmethod
    def set_symbol(cls, symbol: str) -> None:
        """Set financial instrument symbol."""
        cls._local.symbol = symbol

    @classmethod
    def get_symbol(cls) -> Optional[str]:
        """Get financial instrument symbol."""
        return getattr(cls._local, 'symbol', None)

    @classmethod
    def set_strategy_id(cls, strategy_id: str) -> None:
        """Set trading strategy identifier."""
        cls._local.strategy_id = strategy_id

    @classmethod
    def get_strategy_id(cls) -> Optional[str]:
        """Get trading strategy identifier."""
        return getattr(cls._local, 'strategy_id', None)

    @classmethod
    def set_timeframe(cls, timeframe: str) -> None:
        """Set market data timeframe."""
        cls._local.timeframe = timeframe

    @classmethod
    def get_timeframe(cls) -> Optional[str]:
        """Get market data timeframe."""
        return getattr(cls._local, 'timeframe', None)

    @classmethod
    def set_model_version(cls, model_version: str) -> None:
        """Set ML model version."""
        cls._local.model_version = model_version

    @classmethod
    def get_model_version(cls) -> Optional[str]:
        """Get ML model version."""
        return getattr(cls._local, 'model_version', None)

    @classmethod
    def set_prediction_confidence(cls, confidence: float) -> None:
        """Set model prediction confidence score."""
        cls._local.prediction_confidence = confidence

    @classmethod
    def get_prediction_confidence(cls) -> Optional[float]:
        """Get model prediction confidence score."""
        return getattr(cls._local, 'prediction_confidence', None)

    @classmethod
    def set_trade_id(cls, trade_id: str) -> None:
        """Set trade execution identifier."""
        cls._local.trade_id = trade_id

    @classmethod
    def get_trade_id(cls) -> Optional[str]:
        """Get trade execution identifier."""
        return getattr(cls._local, 'trade_id', None)

    # Batch context management
    @classmethod
    def from_trading_context(cls, trading_context: TradingContext) -> None:
        """
        Set logging context from TradingContext object.

        This method extracts available information from a TradingContext
        and sets the corresponding thread-local context fields.

        Args:
            trading_context: TradingContext instance with context information
        """
        cls.set_correlation_id(trading_context.correlation_id)
        cls.set_event_id(trading_context.event_id)
        cls.set_symbol(trading_context.symbol)

        # Set optional fields only if present
        if trading_context.strategy_id:
            cls.set_strategy_id(trading_context.strategy_id)
        if trading_context.timeframe:
            cls.set_timeframe(trading_context.timeframe)
        if trading_context.model_version:
            cls.set_model_version(trading_context.model_version)
        if trading_context.prediction_confidence is not None:
            cls.set_prediction_confidence(trading_context.prediction_confidence)
        if trading_context.trade_id:
            cls.set_trade_id(trading_context.trade_id)

    @classmethod
    def get_available_context(cls) -> Dict[str, Any]:
        """
        Get all available context fields as dictionary.

        Only returns fields that have been set (non-None values) to avoid
        cluttering logs with empty context fields.

        Returns:
            Dictionary with available context information
        """
        context = {}

        # Define all possible context fields with their getters
        context_fields = [
            ('correlation_id', cls.get_correlation_id),
            ('event_id', cls.get_event_id),
            ('symbol', cls.get_symbol),
            ('strategy_id', cls.get_strategy_id),
            ('timeframe', cls.get_timeframe),
            ('model_version', cls.get_model_version),
            ('prediction_confidence', cls.get_prediction_confidence),
            ('trade_id', cls.get_trade_id)
        ]

        # Add only non-None values to context
        for field_name, getter in context_fields:
            value = getter()
            if value is not None:
                context[field_name] = value

        return context

    @classmethod
    def clear(cls) -> None:
        """Clear all context information for current thread."""
        context_attrs = [
            'correlation_id', 'event_id', 'symbol', 'strategy_id',
            'timeframe', 'model_version', 'prediction_confidence', 'trade_id'
        ]

        for attr in context_attrs:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)

    @classmethod
    def generate_new_correlation_id(cls) -> str:
        """
        Generate and set a new correlation ID.

        Returns:
            The generated correlation ID
        """
        correlation_id = f"trade-{uuid.uuid4()}"
        cls.set_correlation_id(correlation_id)
        return correlation_id

    @classmethod
    def generate_new_event_id(cls, service_name: str) -> str:
        """
        Generate and set a new event ID.

        Args:
            service_name: Name of the service generating the event

        Returns:
            The generated event ID
        """
        event_id = f"{service_name}-{uuid.uuid4()}"
        cls.set_event_id(event_id)
        return event_id
