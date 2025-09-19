"""
Example service showing how to use the new SQLAlchemy-based market data access.

This demonstrates the integration pattern where preprocessing service can access
market data through the shared reader port while maintaining hexagonal architecture.
"""

import logging
from datetime import datetime, timedelta
from typing import List, TYPE_CHECKING

from injector import inject

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_core.common.model.data_availability_summary import DataAvailabilitySummary

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)


@inject
class MarketDataPreprocessingService:
    """
    Example service demonstrating market data access for preprocessing.

    Shows how services can use the shared MarketDataReaderPort to access
    market data without violating hexagonal architecture principles.
    """

    def __init__(self, market_data_reader: MarketDataReaderPort):
        """
        Initialize with market data reader port.

        Args:
            market_data_reader: Market data reader port (implemented by adapter)
        """
        self.market_data_reader = market_data_reader
        self.logger = logging.getLogger(__name__)

    def prepare_training_dataset(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        lookback_days: int = 30
    ) -> List[MarketDataModel]:
        """
        Prepare training dataset for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe enum (e.g., Timeframe.ONE_HOUR, Timeframe.ONE_DAY)
            lookback_days: Number of days to look back for training data

        Returns:
            List[MarketDataModel]: Combined market data for training
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        self.logger.info(
            f"Preparing training dataset for {len(symbols)} symbols "
            f"from {start_time} to {end_time}"
        )

        # Use bulk retrieval for efficiency
        market_data = self.market_data_reader.get_multiple_symbols_data_range(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        if not market_data:
            self.logger.warning("No market data found for specified criteria")
            return market_data

        self.logger.info(f"Retrieved {len(market_data)} records for training")
        return market_data

    def get_feature_warmup_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        warmup_periods: int = 100
    ) -> List[MarketDataModel]:
        """
        Get recent data for feature warm-up before inference.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum
            warmup_periods: Number of periods needed for feature calculation

        Returns:
            List[MarketDataModel]: Recent market data for warm-up
        """
        # Calculate approximate time range based on timeframe
        # This is a simplified calculation - real implementation would be more sophisticated
        period_minutes = self._timeframe_to_minutes(timeframe)
        total_minutes = warmup_periods * period_minutes

        start_time = datetime.now() - timedelta(minutes=total_minutes * 2)  # Buffer for weekends/gaps
        end_time = datetime.now()

        self.logger.info(f"Getting warmup data for {symbol}:{timeframe.value}")

        warmup_data = self.market_data_reader.get_symbol_data_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        # Take the most recent N periods
        if len(warmup_data) > warmup_periods:
            warmup_data = warmup_data[-warmup_periods:]  # Take last N elements

        self.logger.info(f"Retrieved {len(warmup_data)} periods for warmup")
        return warmup_data

    def get_live_prices_for_inference(self, symbols: List[str], timeframe: Timeframe) -> List[MarketDataModel]:
        """
        Get latest prices for real-time inference.

        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe enum

        Returns:
            List[MarketDataModel]: Latest market data
        """
        self.logger.info(f"Getting live prices for {len(symbols)} symbols")

        latest_prices = self.market_data_reader.get_latest_prices(
            symbols=symbols,
            timeframe=timeframe
        )

        self.logger.info(f"Retrieved latest prices for {len(latest_prices)} symbols")
        return latest_prices

    def analyze_data_availability(self) -> List[DataAvailabilitySummary]:
        """
        Analyze what market data is available in the system.

        Returns:
            List[DataAvailabilitySummary]: Data availability summary
        """
        self.logger.info("Analyzing market data availability")

        availability = self.market_data_reader.get_data_availability_summary()

        self.logger.info(
            f"Found data for {len(availability)} symbol-timeframe combinations"
        )
        return availability

    def _timeframe_to_minutes(self, timeframe: Timeframe) -> int:
        """
        Convert timeframe enum to minutes.

        Args:
            timeframe: Timeframe enum value

        Returns:
            int: Number of minutes in the timeframe
        """
        timeframe_minutes = {
            Timeframe.MINUTE_1: 1,
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.MINUTE_30: 30,
            Timeframe.HOUR_1: 60,
            Timeframe.HOUR_4: 240,
            Timeframe.DAY_1: 1440,
            Timeframe.WEEK_1: 10080,
        }

        return timeframe_minutes.get(timeframe, 60)  # Default to 1 hour if unknown

    def models_to_dataframe(self, market_data: List[MarketDataModel]) -> 'DataFrame':
        """
        Convert MarketDataModel list to DataFrame for legacy compatibility.

        Args:
            market_data: List of market data models

        Returns:
            DataFrame: Pandas DataFrame with market data
        """
        from pandas import DataFrame

        if not market_data:
            return DataFrame(columns=['symbol', 'timestamp', 'open_price', 'high_price',
                                     'low_price', 'close_price', 'volume', 'timeframe'])

        data = []
        for model in market_data:
            data.append({
                'symbol': model.symbol,
                'timestamp': model.timestamp,
                'open_price': model.open_price,
                'high_price': model.high_price,
                'low_price': model.low_price,
                'close_price': model.close_price,
                'volume': model.volume,
                'timeframe': model.timeframe.value
            })

        df = DataFrame(data)
        return df.set_index(['symbol', 'timestamp']) if len(market_data) > 1 else df.set_index('timestamp')
