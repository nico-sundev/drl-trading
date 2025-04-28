from typing import Any, Optional

import yfinance as yf
from pandas import DataFrame

from ai_trading.data_import.base_data_import_service import BaseDataImportService
from ai_trading.model.symbol_import_container import SymbolImportContainer

# List of stocks in the Dow Jones 30
# tickers = [
#     'MMM', 'AXP', 'AAPL'
# ]


class YahooDataImportService(BaseDataImportService):

    def __init__(
        self, ticker: str, start_date: Optional[Any], end_date: Optional[Any]
    ) -> None:
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def import_data(self, limit: Optional[int] = None) -> list[SymbolImportContainer]:
        return []

    # Get historical data from Yahoo Finance and save it to dictionary
    def fetch_stock_data(self) -> Any | DataFrame:
        return yf.download(self.ticker, start=self.start_date, end=self.end_date)
