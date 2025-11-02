"""
Message publisher port for resampling service.

This port defines the contract for publishing resampled market data
to messaging infrastructure (Kafka, Redis, etc.) for consumption by ingest service.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel


class StoreResampledDataMessagePublisherPort(ABC):
    """
    Interface for publishing resampled market data messages.

    This port enables the resampling service to publish generated higher timeframe
    data to messaging infrastructure without being coupled to specific implementations.
    """

    @abstractmethod
    def publish_resampled_data(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        resampled_data: Dict[Timeframe, List[MarketDataModel]],
        new_candles_count: Dict[Timeframe, int]
    ) -> None:
        """
        Publish resampled market data for ingestion.

        Args:
            symbol: Trading symbol for the resampled data
            base_timeframe: Original timeframe used for resampling
            resampled_data: Dictionary mapping target timeframes to resampled records
            new_candles_count: Number of newly generated candles per timeframe

        Raises:
            PublishingError: If message publishing fails
            SerializationError: If data cannot be serialized
        """
        pass

    @abstractmethod
    def publish_resampling_error(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        target_timeframes: List[Timeframe],
        error_message: str,
        error_details: Dict[str, str]
    ) -> None:
        """
        Publish resampling error for monitoring and alerting.

        Args:
            symbol: Trading symbol where resampling failed
            base_timeframe: Base timeframe being processed
            target_timeframes: Target timeframes that failed
            error_message: Human-readable error description
            error_details: Additional error context and metadata

        Raises:
            PublishingError: If error message publishing fails
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if messaging infrastructure is healthy and reachable.

        Returns:
            bool: True if messaging system is healthy, False otherwise
        """
        pass
