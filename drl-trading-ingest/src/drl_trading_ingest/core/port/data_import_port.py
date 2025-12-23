"""Port (interface) for data import services."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)


class DataProviderPort(ABC):
    """Port defining the interface for data provider operations.

    Supports both batch historical data import and streaming real-time data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data provider.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._is_initialized = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of this data provider."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize provider-specific resources (connections, auth, etc.).

        Raises:
            Exception: If setup fails
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Clean up provider resources.
        """
        pass

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolImportContainer]:
        """
        Fetch historical batch data for a symbol.

        Args:
            symbol: Symbol to fetch (provider-specific format)
            start_date: Start date for data range
            end_date: End date for data range
            timeframe: Timeframe/interval for data
            limit: Maximum number of records to fetch

        Returns:
            List of SymbolImportContainer objects with historical data
        """
        pass

    @abstractmethod
    def map_symbol(self, internal_symbol: str) -> str:
        """
        Map internal symbol format to provider-specific format.

        Args:
            internal_symbol: Symbol in internal format (e.g., "EURUSD")

        Returns:
            Symbol in provider-specific format (e.g., "EUR/USD", "EURUSD")
        """
        pass

    @abstractmethod
    def start_streaming(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Start streaming real-time data for given symbols.

        Args:
            symbols: List of symbols to stream (internal format)
            callback: Function to call with each data update
        """
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        """
        Stop streaming real-time data.
        """
        pass
