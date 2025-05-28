"""High-level messaging service for trading components."""

import logging
from typing import Any, Callable, Dict

from .transport_interface import (
    DeploymentMode,
    Message,
    TransportFactory,
    TransportInterface,
)

logger = logging.getLogger(__name__)


class TradingMessageBus:
    """High-level message bus for trading system components."""

    def __init__(self, transport: TransportInterface):
        self.transport = transport
        self._handlers: Dict[str, Callable] = {}

    # Data Ingestion Events
    def publish_market_data(
        self, symbol: str, timeframe: str, data: Dict[str, Any]
    ) -> None:
        """Publish new market data."""
        message = Message(
            topic=f"market_data.{symbol}.{timeframe}",
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data,
                "event_type": "market_data_received",
            },
        )
        self.transport.publish(message)

    def subscribe_to_market_data(
        self, symbol: str, timeframe: str, handler: Callable
    ) -> None:
        """Subscribe to market data for specific symbol/timeframe."""
        topic = f"market_data.{symbol}.{timeframe}"
        self.transport.subscribe(topic, lambda msg: handler(msg.payload))

    # Feature Engineering Events
    def publish_features_computed(
        self, symbol: str, timeframe: str, features: Dict[str, Any]
    ) -> None:
        """Publish computed features."""
        message = Message(
            topic=f"features.{symbol}.{timeframe}",
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "features": features,
                "event_type": "features_computed",
            },
        )
        self.transport.publish(message)

    def subscribe_to_features(
        self, symbol: str, timeframe: str, handler: Callable
    ) -> None:
        """Subscribe to computed features."""
        topic = f"features.{symbol}.{timeframe}"
        self.transport.subscribe(topic, lambda msg: handler(msg.payload))

    # Trading Signal Events
    def publish_trading_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Publish trading signal with idempotence."""
        signal_id = f"{symbol}_{signal.get('timestamp', '')}"
        message = Message(
            topic=f"signals.{symbol}",
            payload={
                "signal_id": signal_id,
                "symbol": symbol,
                "signal": signal,
                "event_type": "trading_signal_generated",
            },
            correlation_id=signal_id,  # For idempotence
        )
        self.transport.publish(message)

    def subscribe_to_trading_signals(self, symbol: str, handler: Callable) -> None:
        """Subscribe to trading signals."""
        topic = f"signals.{symbol}"
        self.transport.subscribe(topic, lambda msg: handler(msg.payload))

    # Trade Execution Events
    def publish_trade_executed(
        self, symbol: str, execution_result: Dict[str, Any]
    ) -> None:
        """Publish trade execution result."""
        message = Message(
            topic=f"executions.{symbol}",
            payload={
                "symbol": symbol,
                "execution_result": execution_result,
                "event_type": "trade_executed",
            },
        )
        self.transport.publish(message)

    def subscribe_to_trade_executions(self, symbol: str, handler: Callable) -> None:
        """Subscribe to trade execution results."""
        topic = f"executions.{symbol}"
        self.transport.subscribe(topic, lambda msg: handler(msg.payload))

    # RPC Methods for synchronous operations
    def request_inference(
        self, symbol: str, features: Dict[str, Any], timeout: int = 5
    ) -> Dict[str, Any]:
        """Request ML inference synchronously."""
        message = Message(
            topic=f"inference.request.{symbol}",
            payload={
                "symbol": symbol,
                "features": features,
                "request_type": "inference",
            },
        )

        try:
            reply = self.transport.request_reply(message, timeout)
            return reply.payload
        except TimeoutError:
            logger.error(f"Inference request timed out for {symbol}")
            return {"error": "timeout", "symbol": symbol}

    def handle_inference_requests(
        self, symbol: str, inference_handler: Callable
    ) -> None:
        """Handle inference requests for a symbol."""
        topic = f"inference.request.{symbol}"

        def handle_request(message: Message):
            try:
                # Process inference
                result = inference_handler(message.payload)

                # Send reply
                reply = Message(
                    topic=message.reply_to,
                    payload=result,
                    correlation_id=message.correlation_id,
                )
                self.transport.publish(reply)

            except Exception as e:
                error_reply = Message(
                    topic=message.reply_to,
                    payload={"error": str(e), "symbol": symbol},
                    correlation_id=message.correlation_id,
                )
                self.transport.publish(error_reply)

        self.transport.subscribe(topic, handle_request)

    def start(self) -> None:
        """Start the message bus."""
        self.transport.start()

    def stop(self) -> None:
        """Stop the message bus."""
        self.transport.stop()


class TradingMessageBusFactory:
    """Factory for creating message bus instances."""

    @staticmethod
    def create_message_bus(
        mode: DeploymentMode, **transport_kwargs
    ) -> TradingMessageBus:
        """Create message bus based on deployment mode."""
        transport = TransportFactory.create_transport(mode, **transport_kwargs)
        return TradingMessageBus(transport)
