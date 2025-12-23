"""
Feature computation coordinator service.

Coordinates feature computation across multiple timeframes, handling both
resampled data and direct database fetches when needed. This service separates
data fetching and feature computation concerns from the orchestrator.
"""

import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from injector import inject
from pandas import DataFrame

from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.dto.feature_preprocessing_request import (
    FeaturePreprocessingRequest,
)
from drl_trading_core.core.model.feature_computation_request import (
    FeatureComputationRequest,
)
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_preprocess.core.service.compute.computing_service import (
    FeatureComputingService,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import (
    FeatureComputationCoordinatorConfig,
)

logger = logging.getLogger(__name__)


class FeatureComputationCoordinator:
    """
    Coordinates feature computation across timeframes with intelligent data sourcing.

    This service handles:
    - Feature computation using resampled data when available
    - Direct database fetches for timeframes without resampled data
    - Market data conversion and validation
    - Pagination for memory-efficient data fetching

    Responsibilities:
    - Coordinate data sourcing (resampled vs database)
    - Convert MarketDataModel → DataFrame with validation
    - Orchestrate feature computation per timeframe
    - Prepare computed features for storage (add metadata columns)
    """

    @inject
    def __init__(
        self,
        market_data_reader: MarketDataReaderPort,
        feature_computer: FeatureComputingService,
        config: FeatureComputationCoordinatorConfig,
    ) -> None:
        """
        Initialize the feature computation coordinator.

        Args:
            market_data_reader: Port for reading market data from repository
            feature_computer: Service for computing features from market data
            config: Configuration for pagination and data fetching behavior
        """
        self.market_data_reader = market_data_reader
        self.feature_computer = feature_computer
        self._config = config
        self.logger = logging.getLogger(__name__)

    def compute_features_for_timeframes(
        self,
        request: FeaturePreprocessingRequest,
        features_to_compute: List[FeatureDefinition],
        resampled_data: Dict[Timeframe, DataFrame],
    ) -> Dict[Timeframe, DataFrame]:
        """
        Compute features for all target timeframes.

        Uses resampled data when available, fetches from database otherwise.
        This enables feature computation even when resampling is skipped
        (e.g., force_recompute=True with existing target timeframe data).

        Args:
            request: Original preprocessing request with symbol, timeframes, and time range
            features_to_compute: Features that need computation
            resampled_data: Market data from resampling (may be partial or empty)

        Returns:
            Dictionary mapping timeframes to computed features with metadata columns
        """
        computed_features = {}

        for timeframe in request.target_timeframes:
            # Determine data source: resampled or database
            if timeframe in resampled_data:
                market_data = resampled_data[timeframe]
                data_source = "resampled"
            else:
                # Fetch from database when not in resampled_data
                self.logger.info(
                    f"Timeframe {timeframe.value} not in resampled data. "
                    f"Fetching from database for {request.symbol}"
                )
                market_data = self.fetch_market_data_as_dataframe(
                    symbol=request.symbol,
                    timeframe=timeframe,
                    start_time=request.start_time,
                    end_time=request.end_time,
                )
                data_source = "database"

            if market_data.empty:
                self.logger.warning(
                    f"No market data available for {request.symbol} {timeframe.value}"
                )
                continue

            self.logger.info(
                f"Computing {len(features_to_compute)} features for "
                f"{request.symbol} {timeframe.value} ({len(market_data)} data points, "
                f"source: {data_source})"
            )

            # Compute features for this timeframe using batch computation
            computation_request = FeatureComputationRequest(
                dataset_id=DatasetIdentifier(
                    symbol=request.symbol,
                    timeframe=timeframe,
                ),
                feature_definitions=features_to_compute,
                market_data=market_data,
            )
            features_df = self.feature_computer.compute_batch(computation_request)

            if not features_df.empty:
                # Add required columns for feature store
                # event_timestamp: Required by Feast for temporal operations
                # symbol: Required as entity key for feature lookup
                features_df = features_df.copy()
                features_df["event_timestamp"] = pd.to_datetime(features_df.index)
                features_df["symbol"] = request.symbol

                computed_features[timeframe] = features_df
                self.logger.info(
                    f"Computed {len(features_df.columns) - 2} feature columns "  # -2 for event_timestamp and symbol
                    f"for {len(features_df)} data points on {timeframe.value}"
                )
            else:
                self.logger.warning(f"No features computed for {timeframe.value}")

        return computed_features

    def fetch_market_data_as_dataframe(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
    ) -> DataFrame:
        """
        Fetch market data from database with pagination and convert to DataFrame.

        Handles both incremental updates and large time ranges by fetching data
        in configurable chunks. Validates OHLCV data quality during fetch.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            DataFrame with OHLCV columns (capitalized for pandas_ta compatibility)
        """
        all_data = []
        offset = 0
        chunk_size = self._config.pagination_chunk_size or 10000

        self.logger.debug(
            f"Fetching {symbol} {timeframe.value} data from {start_time} to {end_time} "
            f"(chunk_size={chunk_size})"
        )

        while True:
            try:
                # Use paginated data access
                chunk = self.market_data_reader.get_symbol_data_range_paginated(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    limit=chunk_size,
                    offset=offset,
                )

                if not chunk:
                    break

                # Validate data quality
                valid_chunk = [
                    record for record in chunk if not self._is_invalid_ohlcv(record)
                ]
                all_data.extend(valid_chunk)

                self.logger.debug(
                    f"Fetched chunk: {len(chunk)} records, "
                    f"valid: {len(valid_chunk)}, total: {len(all_data)}"
                )

                # If we got less than chunk_size, we're done
                if len(chunk) < chunk_size:
                    break

                offset += chunk_size

            except AttributeError:
                # Fallback to non-paginated method if not implemented
                self.logger.warning(
                    f"Pagination not available, falling back to full range query for {symbol}"
                )
                all_data = self.market_data_reader.get_symbol_data_range(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )
                # Validate fallback data
                all_data = [
                    record for record in all_data if not self._is_invalid_ohlcv(record)
                ]
                break

        self.logger.info(
            f"Fetched {len(all_data)} valid records for {symbol} {timeframe.value}"
        )

        return self._convert_to_dataframe(all_data)

    def _convert_to_dataframe(
        self, market_data_list: List[MarketDataModel]
    ) -> DataFrame:
        """
        Convert MarketDataModel list to DataFrame.

        Uses capitalized column names (Open, High, Low, Close, Volume) for
        pandas_ta compatibility.

        Args:
            market_data_list: List of market data models

        Returns:
            DataFrame with timestamp index and OHLCV columns
        """
        if not market_data_list:
            # Return empty DataFrame with correct structure
            df = DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            df.index.name = "timestamp"
            return df

        df_data = []
        for market_data in market_data_list:
            df_data.append(
                {
                    "timestamp": market_data.timestamp,
                    "Open": market_data.open_price,
                    "High": market_data.high_price,
                    "Low": market_data.low_price,
                    "Close": market_data.close_price,
                    "Volume": market_data.volume,
                }
            )

        df = DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        return df

    def _is_invalid_ohlcv(self, record: MarketDataModel) -> bool:
        """
        Check if a market data record has invalid OHLCV relationships.

        Invalid conditions:
        - High price < Low price
        - Open/Close outside High/Low range
        - Negative volume
        - Zero or negative prices

        Args:
            record: Market data record to validate

        Returns:
            True if the record has invalid OHLCV data
        """
        try:
            # Check for negative or zero prices
            if (
                record.open_price <= 0
                or record.high_price <= 0
                or record.low_price <= 0
                or record.close_price <= 0
            ):
                return True

            # Check high < low (impossible market condition)
            if record.high_price < record.low_price:
                return True

            # Check if open/close are outside high/low range
            if (
                record.open_price > record.high_price
                or record.open_price < record.low_price
                or record.close_price > record.high_price
                or record.close_price < record.low_price
            ):
                return True

            # Check for negative volume
            if record.volume < 0:
                return True

            return False

        except (AttributeError, TypeError):
            # If any price/volume field is missing or invalid type
            return True
