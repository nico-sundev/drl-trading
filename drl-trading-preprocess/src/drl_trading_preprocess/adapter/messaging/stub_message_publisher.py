"""
Stub implementation of message publisher for development and testing.

This adapter provides a non-functional stub implementation of the MessagePublisherPort
that logs messages instead of actually publishing them. This enables development
and testing of the resampling service without requiring Kafka infrastructure.
"""

import logging
from typing import Dict, List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.port.message_publisher_port import MessagePublisherPort


class StubMessagePublisher(MessagePublisherPort):
    """
    Stub message publisher for development and testing.

    This implementation logs all publishing operations instead of actually
    sending messages to messaging infrastructure. Useful for:
    - Local development without Kafka
    - Unit testing resampling logic
    - Integration testing preprocessing pipeline
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize stub publisher with configurable logging.

        Args:
            log_level: Logging level for published messages (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self._published_messages: List[Dict] = []
        self._error_messages: List[Dict] = []
        self._is_healthy = True

    def publish_resampled_data(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        resampled_data: Dict[Timeframe, List[MarketDataModel]],
        new_candles_count: Dict[Timeframe, int]
    ) -> None:
        """
        Log resampled data publication (stub implementation).

        Stores message details for inspection and logs summary information.
        """
        total_records = sum(len(records) for records in resampled_data.values())
        total_new_candles = sum(new_candles_count.values())

        message = {
            "type": "resampled_data",
            "symbol": symbol,
            "base_timeframe": base_timeframe.value,
            "target_timeframes": [tf.value for tf in resampled_data.keys()],
            "total_records": total_records,
            "new_candles_count": dict(
                (tf.value, count) for tf, count in new_candles_count.items()
            ),
            "total_new_candles": total_new_candles
        }

        self._published_messages.append(message)

        self.logger.info(
            f"[STUB] Published resampled data: {symbol} "
            f"({base_timeframe.value} → {list(resampled_data.keys())}) "
            f"- {total_records} records, {total_new_candles} new candles"
        )

        # Log detailed breakdown per timeframe
        for timeframe, records in resampled_data.items():
            new_count = new_candles_count.get(timeframe, 0)
            self.logger.debug(
                f"[STUB] {symbol}:{timeframe.value} - "
                f"{len(records)} total records, {new_count} new candles"
            )

    def publish_resampling_error(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        target_timeframes: List[Timeframe],
        error_message: str,
        error_details: Dict[str, str]
    ) -> None:
        """
        Log resampling error (stub implementation).

        Stores error details for inspection and logs error information.
        """
        error_msg = {
            "type": "resampling_error",
            "symbol": symbol,
            "base_timeframe": base_timeframe.value,
            "target_timeframes": [tf.value for tf in target_timeframes],
            "error_message": error_message,
            "error_details": error_details
        }

        self._error_messages.append(error_msg)

        self.logger.error(
            f"[STUB] Resampling error for {symbol} "
            f"({base_timeframe.value} → {[tf.value for tf in target_timeframes]}): "
            f"{error_message}"
        )

        if error_details:
            self.logger.error(f"[STUB] Error details: {error_details}")

    def health_check(self) -> bool:
        """
        Return stub health status.

        Always returns True unless explicitly set to unhealthy for testing.
        """
        self.logger.debug("[STUB] Health check - stub is always healthy")
        return self._is_healthy

    # Additional methods for testing and inspection

    def get_published_messages(self) -> List[Dict]:
        """Get all published messages for testing inspection."""
        return self._published_messages.copy()

    def get_error_messages(self) -> List[Dict]:
        """Get all error messages for testing inspection."""
        return self._error_messages.copy()

    def clear_messages(self) -> None:
        """Clear all stored messages (useful for test cleanup)."""
        self._published_messages.clear()
        self._error_messages.clear()

    def set_health_status(self, is_healthy: bool) -> None:
        """Set health status for testing failure scenarios."""
        self._is_healthy = is_healthy
        self.logger.info(f"[STUB] Health status set to: {is_healthy}")

    def get_message_count(self) -> Dict[str, int]:
        """Get count of different message types."""
        return {
            "resampled_data": len(self._published_messages),
            "errors": len(self._error_messages),
            "total": len(self._published_messages) + len(self._error_messages)
        }

    def get_symbols_processed(self) -> List[str]:
        """Get list of symbols that have been processed."""
        symbols = set()
        for msg in self._published_messages:
            symbols.add(msg["symbol"])
        for msg in self._error_messages:
            symbols.add(msg["symbol"])
        return sorted(list(symbols))
