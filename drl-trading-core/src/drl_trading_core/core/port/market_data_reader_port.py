"""
Market data reader port for shared read-only access.

This port defines the contract for reading market data across services.
It provides read-only access to the market data platform without write operations,
supporting bulk data retrieval for preprocessing and real-time access for inference.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_core.common.model.data_availability_summary import DataAvailabilitySummary


class MarketDataReaderPort(ABC):
    """
    Interface for read-only market data access.

    This port enables multiple services to read market data from the shared
    market data platform while maintaining clear separation from write operations
    that remain exclusive to the ingest service.
    """

    @abstractmethod
    def get_symbol_data_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataModel]:
        """
        Get market data for a symbol within a time range.

        Optimized for bulk data retrieval needed by preprocessing services.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Data timeframe enum
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List[MarketDataModel]: Market data sorted by timestamp ascending

        Raises:
            ValueError: If symbol or timeframe is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_symbol_data_range_paginated(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        limit: int,
        offset: int = 0
    ) -> List[MarketDataModel]:
        """
        Get paginated market data for a symbol within a time range.

        Enables memory-efficient processing of large datasets by fetching
        data in manageable chunks. Essential for stateful resampling services.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Data timeframe enum
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)

        Returns:
            List[MarketDataModel]: Market data sorted by timestamp ascending

        Raises:
            ValueError: If symbol or timeframe is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_multiple_symbols_data_range(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataModel]:
        """
        Get market data for multiple symbols within a time range.

        Efficient bulk retrieval for portfolio-level analysis and training.

        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe enum
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List[MarketDataModel]: Market data sorted by symbol then timestamp

        Raises:
            ValueError: If any symbol or timeframe is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_latest_prices(
        self,
        symbols: List[str],
        timeframe: Timeframe
    ) -> List[MarketDataModel]:
        """
        Get the most recent price data for multiple symbols.

        Optimized for real-time inference and live trading decisions.

        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe enum

        Returns:
            List[MarketDataModel]: Latest market data for each symbol

        Raises:
            ValueError: If any symbol or timeframe is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_data_availability(
        self,
        symbol: str,
        timeframe: Timeframe
    ) -> DataAvailabilitySummary:
        """
        Get data availability summary for a specific symbol and timeframe.

        Performs efficient database aggregation (COUNT, MIN, MAX) to determine
        data availability without fetching actual records. This is 100-1000x faster
        than fetching all data for availability checking.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum

        Returns:
            DataAvailabilitySummary: Availability summary for the symbol

        Raises:
            ValueError: If symbol or timeframe is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_symbol_available_timeframes(self, symbol: str) -> List[Timeframe]:
        """
        Get all available timeframes for a symbol.

        Useful for discovering what data is available before fetching.

        Args:
            symbol: Trading symbol

        Returns:
            List[Timeframe]: Available timeframes for the symbol

        Raises:
            ValueError: If symbol is invalid
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_data_availability_summary(self) -> List[DataAvailabilitySummary]:
        """
        Get summary of data availability across all symbols and timeframes.

        Returns overview of what data is available for system monitoring
        and planning purposes.

        Returns:
            List[DataAvailabilitySummary]: Summary of available data

        Raises:
            DatabaseConnectionError: If database access fails
        """
        pass
