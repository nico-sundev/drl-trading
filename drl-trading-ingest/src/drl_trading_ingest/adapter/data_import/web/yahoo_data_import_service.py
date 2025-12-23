"""Service to import OHLC data from Yahoo Finance."""

import logging
from typing import Any, Callable, Dict, List, Optional

import yfinance as yf
from drl_trading_common.config.local_data_import_config import LocalDataImportConfig
from pandas import DataFrame

from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_ingest.core.port import DataProviderPort

logger = logging.getLogger(__name__)


class YahooDataImportService(DataProviderPort):
    """Service to import OHLC data from Yahoo Finance."""

    PROVIDER_NAME = "yahoo"

    def __init__(
        self,
        config: Dict[str, Any],
        ticker: Optional[str] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Yahoo data import service.

        Args:
            config: Yahoo-specific configuration
            ticker: Stock ticker symbol (optional, can be in config)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        super().__init__(config)

        # Support both dict and LocalDataImportConfig
        if isinstance(config, LocalDataImportConfig):
            self.ticker = ticker
        else:
            self.ticker = ticker or config.get("symbols", ["AAPL"])[0]

        self.start_date = start_date
        self.end_date = end_date

    @property
    def provider_name(self) -> str:
        """Return the unique name of this data provider."""
        return self.PROVIDER_NAME

    def setup(self) -> None:
        """Initialize Yahoo Finance (no auth required for public data)."""
        logger.info(f"Setting up {self.provider_name} provider...")
        self._is_initialized = True
        logger.info(f"{self.provider_name} provider ready")

    def teardown(self) -> None:
        """Clean up Yahoo Finance resources."""
        logger.info(f"Tearing down {self.provider_name} provider...")
        self._is_initialized = False

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolImportContainer]:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Interval (not used by yfinance in this implementation)
            limit: Not used

        Returns:
            List of SymbolImportContainer objects
        """
        logger.info(
            f"Fetching historical data from {self.provider_name}: "
            f"symbol={symbol}, start={start_date}, end={end_date}"
        )

        yahoo_symbol = self.map_symbol(symbol)
        _ = self.fetch_stock_data(yahoo_symbol, start_date, end_date)

        # TODO: Convert DataFrame to SymbolImportContainer
        logger.warning(f"{self.provider_name}: Stub implementation - returning empty containers")
        return []

    def map_symbol(self, internal_symbol: str) -> str:
        """
        Map internal symbol format to Yahoo Finance format.

        Args:
            internal_symbol: Symbol in internal format

        Returns:
            Symbol in Yahoo Finance format (typically same)
        """
        return internal_symbol.upper()

    def start_streaming(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Yahoo Finance does not support streaming via yfinance library."""
        logger.warning(f"{self.provider_name} provider does not support streaming")

    def stop_streaming(self) -> None:
        """Yahoo Finance does not support streaming."""
        pass

    def fetch_stock_data(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> Any | DataFrame:
        """
        Get historical data from Yahoo Finance.

        Args:
            ticker: Stock ticker (defaults to self.ticker)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with historical stock data
        """
        symbol = ticker or self.ticker
        start = start_date or self.start_date
        end = end_date or self.end_date

        return yf.download(symbol, start=start, end=end)
