"""Twelve Data provider implementation."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_ingest.core.port import DataProviderPort

logger = logging.getLogger(__name__)


class TwelveDataProvider(DataProviderPort):
    """Twelve Data market data provider.

    Supports stocks, forex, crypto, ETFs, and indices data via REST API and WebSocket.
    """

    PROVIDER_NAME = "twelve_data"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Twelve Data provider.

        Args:
            config: Twelve Data-specific configuration
        """
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "TWELVE_DATA_API_KEY"))
        self.base_url = config.get("base_url", "https://api.twelvedata.com")
        self.websocket_url = config.get("websocket_url", "wss://ws.twelvedata.com/v1")
        self._ws_connection = None

    @property
    def provider_name(self) -> str:
        """Return the unique name of this data provider."""
        return self.PROVIDER_NAME

    def setup(self) -> None:
        """
        Initialize Twelve Data API connection.

        Raises:
            Exception: If API key is missing or API is unreachable
        """
        logger.info(f"Setting up {self.provider_name} provider...")

        if not self.api_key:
            raise ValueError(
                f"{self.provider_name}: API key not found. "
                f"Set {self.config.get('api_key_env', 'TWELVE_DATA_API_KEY')} environment variable."
            )

        # TODO: Test API connection with a simple call
        # import requests
        # response = requests.get(
        #     f"{self.base_url}/time_series",
        #     params={"symbol": "AAPL", "interval": "1day", "outputsize": 1, "apikey": self.api_key}
        # )
        # if response.status_code == 401:
        #     raise Exception("Invalid Twelve Data API key")
        # elif response.status_code != 200:
        #     raise Exception(f"Twelve Data API unreachable: {response.status_code}")

        self._is_initialized = True
        logger.info(f"{self.provider_name} provider ready")

    def teardown(self) -> None:
        """Clean up Twelve Data connections."""
        logger.info(f"Tearing down {self.provider_name} provider...")

        if self._ws_connection:
            self.stop_streaming()

        self._is_initialized = False
        logger.info(f"{self.provider_name} provider stopped")

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = "1h",
        limit: Optional[int] = 5000,
    ) -> List[SymbolImportContainer]:
        """
        Fetch historical time series data from Twelve Data.

        Args:
            symbol: Symbol in internal format (e.g., "AAPL", "EUR/USD")
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            timeframe: Interval (1min, 5min, 15min, 1h, 1day, 1week, 1month)
            limit: Number of data points (outputsize, default: 5000)

        Returns:
            List of SymbolImportContainer objects with historical data
        """
        logger.info(
            f"Fetching historical data from {self.provider_name}: "
            f"symbol={symbol}, interval={timeframe}, outputsize={limit}"
        )

        # TODO: Implement actual API call
        # Map symbol to Twelve Data format before API call: self.map_symbol(symbol)
        # Example endpoint: GET /time_series
        # Params: symbol=AAPL, interval=1h, outputsize=5000, apikey=xxx
        #
        # import requests
        # params = {
        #     "symbol": td_symbol,
        #     "interval": self._map_timeframe(timeframe),
        #     "outputsize": limit,
        #     "apikey": self.api_key,
        #     "format": "JSON"
        # }
        # if start_date:
        #     params["start_date"] = start_date
        # if end_date:
        #     params["end_date"] = end_date
        #
        # response = requests.get(f"{self.base_url}/time_series", params=params)
        # data = response.json()
        #
        # # Convert Twelve Data format to SymbolImportContainer
        # # Twelve Data format: {"values": [{"datetime": "...", "open": "...", ...}]}
        # return self._convert_to_containers(data, symbol, timeframe)

        logger.warning(f"{self.provider_name}: Stub implementation - returning empty data")
        return []

    def map_symbol(self, internal_symbol: str) -> str:
        """
        Map internal symbol format to Twelve Data format.

        Args:
            internal_symbol: Symbol in internal format

        Returns:
            Symbol in Twelve Data format

        Examples:
            "AAPL" -> "AAPL" (stocks)
            "EURUSD" -> "EUR/USD" (forex)
            "BTCUSD" -> "BTC/USD" (crypto)
        """
        # Twelve Data uses different formats depending on asset type
        # For forex/crypto, it typically uses "/" separator
        # For stocks, it uses ticker as-is

        # Simple heuristic: if 6+ chars and no special chars, assume forex/crypto
        if len(internal_symbol) >= 6 and internal_symbol.isalnum():
            # Assume format like "EURUSD" -> "EUR/USD"
            base = internal_symbol[:3]
            quote = internal_symbol[3:]
            return f"{base}/{quote}"

        return internal_symbol.upper()

    def start_streaming(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Start streaming real-time market data via Twelve Data WebSocket.

        Args:
            symbols: List of symbols to stream (internal format)
            callback: Function to call with each data update

        Example data format:
            {
                "event": "price",
                "symbol": "AAPL",
                "price": "150.25",
                "timestamp": 1638747600
            }
        """
        logger.info(
            f"Starting {self.provider_name} WebSocket stream for symbols: {symbols}"
        )

        # TODO: Implement WebSocket connection
        # Map symbols to Twelve Data format before connection: [self.map_symbol(s) for s in symbols]
        # Example: wss://ws.twelvedata.com/v1/quotes/price?apikey=xxx
        #
        # import websocket
        # import json
        # import threading
        #
        # def on_open(ws):
        #     # Subscribe to symbols
        #     subscribe_msg = {
        #         "action": "subscribe",
        #         "params": {
        #             "symbols": ",".join(td_symbols)
        #         }
        #     }
        #     ws.send(json.dumps(subscribe_msg))
        #
        # def on_message(ws, message):
        #     data = json.loads(message)
        #     callback(data)
        #
        # ws_url = f"{self.websocket_url}/quotes/price?apikey={self.api_key}"
        #
        # self._ws_connection = websocket.WebSocketApp(
        #     ws_url,
        #     on_open=on_open,
        #     on_message=on_message,
        #     on_error=lambda ws, error: logger.error(f"WebSocket error: {error}"),
        #     on_close=lambda ws: logger.info("WebSocket closed")
        # )
        #
        # ws_thread = threading.Thread(target=self._ws_connection.run_forever)
        # ws_thread.daemon = True
        # ws_thread.start()

        logger.warning(f"{self.provider_name}: Stub implementation - WebSocket not started")

    def stop_streaming(self) -> None:
        """Stop WebSocket streaming."""
        logger.info(f"Stopping {self.provider_name} WebSocket stream")

        if self._ws_connection:
            # TODO: Implement WebSocket close
            # self._ws_connection.close()
            self._ws_connection = None
            logger.info(f"{self.provider_name} WebSocket stream stopped")

    def _map_timeframe(self, internal_timeframe: str) -> str:
        """
        Map internal timeframe to Twelve Data interval format.

        Args:
            internal_timeframe: Internal timeframe (e.g., "1h", "1d")

        Returns:
            Twelve Data interval (e.g., "1h", "1day")
        """
        # Twelve Data uses: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1day",
            "1w": "1week",
        }
        return mapping.get(internal_timeframe, internal_timeframe)
