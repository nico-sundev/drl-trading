import logging
from typing import Dict, Optional

import pandas as pd
from drl_trading_common import TechnicalMetricsServiceInterface
from drl_trading_common.utils import ensure_datetime_index
from pandas import DataFrame

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet

logger = logging.getLogger(__name__)

class TechnicalMetricsService(TechnicalMetricsServiceInterface):
    """
    Service that calculates and provides technical metrics like ATR for a specific timeframe.

    This service calculates metrics during initialization and caches them for
    later use to avoid redundant calculations across features.
    """

    def __init__(self, asset_data: AssetPriceDataSet) -> None:
        """
        Initialize the technical metrics service with the asset price dataset.

        Args:
            asset_data: The asset price dataset for a specific timeframe
        """
        self._timeframe = asset_data.timeframe
        self._source_df = ensure_datetime_index(
            asset_data.asset_price_dataset,
            f"Technical metrics source data for {self._timeframe}",
        )

        # Cache for calculated metrics
        self._atr_cache: Dict[int, DataFrame] = {}

        logger.debug(
            f"Initialized TechnicalMetricsService for timeframe {self._timeframe}"
        )

    @property
    def timeframe(self) -> str:
        """Get the timeframe this metrics service is associated with."""
        return self._timeframe

    def get_atr(self, period: int = 14) -> Optional[DataFrame]:
        """
        Get Average True Range (ATR) values for the current timeframe.

        ATR is calculated using the standard formula:
        1. Calculate True Range (TR):
           - Method 1: Current High - Current Low
           - Method 2: |Current High - Previous Close|
           - Method 3: |Current Low - Previous Close|
           - TR is the maximum of these three methods
        2. ATR is typically the simple moving average of TR over the specified period

        Args:
            period: The period for ATR calculation, default is 14

        Returns:
            DataFrame: DataFrame with DatetimeIndex and ATR values, or None if calculation fails
        """
        # Return from cache if available
        if period in self._atr_cache:
            return self._atr_cache[period]

        # Calculate ATR and store in cache
        result_df = self._calculate_atr(period)
        if result_df is not None:
            self._atr_cache[period] = result_df

        return result_df

    def _calculate_atr(self, period: int) -> Optional[DataFrame]:
        """
        Calculate Average True Range (ATR) for the given period.

        This is a protected method to allow for testing and mocking.

        Args:
            period: The period for ATR calculation

        Returns:
            DataFrame: DataFrame with DatetimeIndex and ATR values, or None if calculation fails
        """
        try:
            df = self._source_df.copy()

            # Check if required columns exist
            required_columns = ["High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                logger.error(
                    f"Missing required columns for ATR calculation: {required_columns}"
                )
                return None

            # Calculate True Range
            high_low = df["High"] - df["Low"]
            high_prev_close = pd.Series(
                abs(df["High"] - df["Close"].shift(1)), index=df.index
            )
            low_prev_close = pd.Series(
                abs(df["Low"] - df["Close"].shift(1)), index=df.index
            )

            true_range = pd.DataFrame(index=df.index)
            true_range["TR"] = pd.concat(
                [high_low, high_prev_close, low_prev_close], axis=1
            ).max(axis=1)

            # Calculate ATR (simple moving average of TR)
            true_range["ATR"] = true_range["TR"].rolling(window=period).mean()

            # Prepare result DataFrame
            result_df = true_range[["ATR"]].copy()
            result_df.index.name = "Time"

            return result_df

        except Exception as e:
            logger.error(
                f"Error calculating ATR with period {period}: {e}", exc_info=True
            )
            return None


class TechnicalMetricsServiceFactory:
    """
    Factory for creating TechnicalMetricsService instances.
    """

    @staticmethod
    def create(asset_data: AssetPriceDataSet) -> TechnicalMetricsServiceInterface:
        """
        Create a new TechnicalMetricsService instance for the given asset data.

        Args:
            asset_data: The asset price dataset

        Returns:
            TechnicalMetricsServiceInterface: A metrics service instance
        """
        return TechnicalMetricsService(asset_data)
