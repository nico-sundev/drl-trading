from typing import List, Optional
from pandas import DataFrame
import yfinance as yf
from IPython.display import display

from ai_trading.data_import.base_data_import_service import (
    BaseDataImportService,
)
from ai_trading.model.split_dataset_container import SplitDataSetContainer

# List of stocks in the Dow Jones 30
# tickers = [
#     'MMM', 'AXP', 'AAPL'
# ]


class YahooDataImportService(BaseDataImportService):

    def __init__(self, ticker: str, start_date, end_date):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def import_data(self, limit: Optional[int] = None) -> DataFrame:
        return self.fetch_stock_data()

    # Get historical data from Yahoo Finance and save it to dictionary
    def fetch_stock_data(self):
        return yf.download(self.ticker, start=self.start_date, end=self.end_date)
