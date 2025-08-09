"""
Trading event payload structures for Kafka message standardization.

Provides standardized payload structures for cross-service communication
in the DRL trading microservice ecosystem.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from drl_trading_common.model.trading_context import TradingContext


class TradingEventPayload(BaseModel):
    """
    Standard payload structure for all trading events in Kafka messages.

    Combines business data with context information for cross-service communication.
    """

    # Business data (service-specific)
    data: Dict[str, Any] = Field(description="Service-specific business data")

    # Context information (standardized across services)
    context: TradingContext = Field(description="Cross-service context information")

    # Message metadata
    message_type: str = Field(description="Type of message (e.g., market_data, prediction, trade_signal)")
    version: str = Field(default="1.0.0", description="Message schema version")

    @classmethod
    def create_market_data_payload(cls, symbol: str, price: float, volume: float,
                                 timestamp: datetime, **metadata: Any) -> "TradingEventPayload":
        """
        Create payload for market data events.

        Args:
            symbol: Financial instrument symbol
            price: Market price
            volume: Trading volume
            timestamp: Data timestamp
            **metadata: Additional metadata

        Returns:
            TradingEventPayload for market data
        """
        context = TradingContext.create_initial_context(symbol, **metadata)

        return cls(
            data={
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'timestamp': timestamp.isoformat()
            },
            context=context,
            message_type='market_data'
        )

    @classmethod
    def create_prediction_payload(cls, context: TradingContext, prediction: float,
                                confidence: float, **metadata: Any) -> "TradingEventPayload":
        """
        Create payload for prediction events.

        Args:
            context: Existing trading context
            prediction: Model prediction value
            confidence: Prediction confidence score
            **metadata: Additional metadata

        Returns:
            TradingEventPayload for predictions
        """
        # Enrich context with model information
        context.enrich_with_model(
            model_version=metadata.get('model_version', 'unknown'),
            prediction_confidence=confidence
        )

        return cls(
            data={
                'prediction': prediction,
                'confidence': confidence,
                **metadata
            },
            context=context,
            message_type='prediction'
        )

    @classmethod
    def create_trade_signal_payload(cls, context: TradingContext, action: str,
                                  quantity: float, price: Optional[float] = None,
                                  **metadata: Any) -> "TradingEventPayload":
        """
        Create payload for trade signal events.

        Args:
            context: Existing trading context
            action: Trade action (buy/sell/hold)
            quantity: Trade quantity
            price: Target price (optional)
            **metadata: Additional metadata

        Returns:
            TradingEventPayload for trade signals
        """
        return cls(
            data={
                'action': action,
                'quantity': quantity,
                'price': price,
                **metadata
            },
            context=context,
            message_type='trade_signal'
        )
