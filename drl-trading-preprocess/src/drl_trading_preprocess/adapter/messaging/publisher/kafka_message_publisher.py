"""
Kafka implementation of message publisher for resampled market data.

This adapter uses KafkaProducerAdapter to publish resampled market data
to Kafka topics with retry logic and resilience behavior.
"""

import logging
from typing import Dict, List

from drl_trading_common.adapter.messaging.kafka_producer_adapter import (
    KafkaProducerAdapter,
)
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.port.message_publisher_port import StoreResampledDataMessagePublisherPort


class KafkaMessagePublisher(StoreResampledDataMessagePublisherPort):
    """
    Kafka-based message publisher for resampled market data.

    This implementation publishes resampled data and errors to Kafka topics
    using the KafkaProducerAdapter for retry logic and resilience.

    Attributes:
        _resampled_data_producer: Producer for resampled data events.
        _error_producer: Producer for error events (DLQ).
        _resampled_data_topic: Topic name for resampled data.
        _error_topic: Topic name for errors (DLQ).
    """

    def __init__(
        self,
        resampled_data_producer: KafkaProducerAdapter,
        error_producer: KafkaProducerAdapter,
        resampled_data_topic: str,
        error_topic: str,
    ) -> None:
        """
        Initialize Kafka message publisher with producers and topic names.

        Args:
            resampled_data_producer: Producer for resampled data (with appropriate retry config).
            error_producer: Producer for errors (typically minimal retry for DLQ).
            resampled_data_topic: Topic name for publishing resampled data.
            error_topic: Topic name for publishing errors (DLQ).
        """
        self.logger = logging.getLogger(__name__)
        self._resampled_data_producer = resampled_data_producer
        self._error_producer = error_producer
        self._resampled_data_topic = resampled_data_topic
        self._error_topic = error_topic

        self.logger.info(
            f"KafkaMessagePublisher initialized: "
            f"resampled_data_topic={resampled_data_topic}, "
            f"error_topic={error_topic}"
        )

    def publish_resampled_data(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        resampled_data: Dict[Timeframe, List[MarketDataModel]],
        new_candles_count: Dict[Timeframe, int],
    ) -> None:
        """
        Publish resampled market data to Kafka.

        Serializes the resampled data to JSON and publishes to the configured
        topic. Uses symbol as the message key for partitioning consistency.

        Args:
            symbol: Trading symbol for the resampled data.
            base_timeframe: Original timeframe used for resampling.
            resampled_data: Dictionary mapping target timeframes to resampled records.
            new_candles_count: Number of newly generated candles per timeframe.

        Raises:
            KafkaException: If publishing fails after all retries.
        """
        total_records = sum(len(records) for records in resampled_data.values())
        total_new_candles = sum(new_candles_count.values())

        # Prepare payload - convert models to dict for JSON serialization
        payload = {
            "symbol": symbol,
            "base_timeframe": base_timeframe.value,
            "resampled_data": {
                tf.value: [record.model_dump() for record in records]
                for tf, records in resampled_data.items()
            },
            "new_candles_count": {
                tf.value: count for tf, count in new_candles_count.items()
            },
            "total_records": total_records,
            "total_new_candles": total_new_candles,
        }

        # Publish with retry logic from KafkaProducerAdapter
        self._resampled_data_producer.publish(
            topic=self._resampled_data_topic,
            key=symbol,  # Partition by symbol for ordering
            value=payload,
            headers={
                "event_type": "resampled_data",
                "symbol": symbol,
                "base_timeframe": base_timeframe.value,
            },
        )

        self.logger.info(
            f"Published resampled data: {symbol} "
            f"({base_timeframe.value} → {list(resampled_data.keys())}) "
            f"- {total_records} records, {total_new_candles} new candles"
        )

    def publish_resampling_error(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        target_timeframes: List[Timeframe],
        error_message: str,
        error_details: Dict[str, str],
    ) -> None:
        """
        Publish resampling error to Kafka DLQ.

        Serializes error information to JSON and publishes to the error topic.
        Uses symbol as the message key for consistency.

        Args:
            symbol: Trading symbol where resampling failed.
            base_timeframe: Base timeframe being processed.
            target_timeframes: Target timeframes that failed.
            error_message: Human-readable error description.
            error_details: Additional error context and metadata.

        Raises:
            KafkaException: If publishing to DLQ fails after all retries.
        """
        payload = {
            "symbol": symbol,
            "base_timeframe": base_timeframe.value,
            "target_timeframes": [tf.value for tf in target_timeframes],
            "error_message": error_message,
            "error_details": error_details,
        }

        # Publish to DLQ (typically has minimal retry config)
        self._error_producer.publish(
            topic=self._error_topic,
            key=symbol,
            value=payload,
            headers={
                "event_type": "resampling_error",
                "symbol": symbol,
                "base_timeframe": base_timeframe.value,
            },
        )

        self.logger.error(
            f"Published resampling error for {symbol} "
            f"({base_timeframe.value} → {[tf.value for tf in target_timeframes]}): "
            f"{error_message}"
        )

    def health_check(self) -> bool:
        """
        Check if Kafka producers are healthy.

        Returns:
            True if both producers are healthy, False otherwise.
        """
        resampled_healthy = self._resampled_data_producer.health_check()
        error_healthy = self._error_producer.health_check()

        is_healthy = resampled_healthy and error_healthy

        if not is_healthy:
            self.logger.warning(
                f"Health check failed: "
                f"resampled_data_producer={resampled_healthy}, "
                f"error_producer={error_healthy}"
            )

        return is_healthy

    def close(self) -> None:
        """Close both Kafka producers and flush pending messages."""
        self.logger.info("Closing KafkaMessagePublisher...")
        self._resampled_data_producer.close()
        self._error_producer.close()
        self.logger.info("KafkaMessagePublisher closed successfully")
