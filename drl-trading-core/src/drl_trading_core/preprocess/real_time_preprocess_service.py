"""Real-time preprocessing service for inference support."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
from drl_trading_common.config.feature_config import FeaturesConfig
from injector import inject
from pandas import DataFrame, Series

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.feast.feast_service import FeastServiceInterface
from drl_trading_core.preprocess.feature.real_time_feature_aggregator import (
    RealTimeFeatureAggregatorInterface,
)

logger = logging.getLogger(__name__)


class RealTimePreprocessServiceInterface(ABC):
    """
    Interface for real-time preprocessing during inference.

    This service handles single-record feature computation for real-time
    inference scenarios.
    """

    @abstractmethod
    def initialize_for_symbol(
        self, symbol: str, timeframe: str, historical_data: DataFrame
    ) -> bool:
        """
        Initialize the service for a specific symbol with historical context.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            historical_data: Historical price data for context

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def compute_features_for_latest_record(
        self, symbol: str, latest_record: Series
    ) -> Dict[str, float]:
        """
        Compute features for the latest price record.

        Args:
            symbol: Trading symbol
            latest_record: Latest price data as pandas Series

        Returns:
            Dict mapping feature names to computed values"""
        pass

    @abstractmethod
    def get_online_features(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve features from online feature store.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp for feature retrieval

        Returns:
            Dict of features or None if not available
        """
        pass

    @abstractmethod
    def get_symbol_status(self, symbol: str) -> Dict:
        """
        Get status information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary containing status information
        """
        pass

    @abstractmethod
    def cleanup_symbol(self, symbol: str) -> None:
        """
        Clean up resources for a symbol.

        Args:
            symbol: Trading symbol
        """
        pass

    @abstractmethod
    def get_all_symbols(self) -> list:
        """
        Get list of all initialized symbols.

        Returns:
            List of symbol names
        """
        pass


class RealTimePreprocessService(RealTimePreprocessServiceInterface):
    """
    Real-time preprocessing service for inference.

    This service manages real-time feature computation and retrieval
    for inference scenarios, maintaining state per symbol.
    """

    @inject
    def __init__(
        self,
        features_config: FeaturesConfig,
        real_time_aggregator: RealTimeFeatureAggregatorInterface,
        feast_service: FeastServiceInterface,
    ) -> None:
        """
        Initialize the real-time preprocessing service.

        Args:
            features_config: Configuration for feature computation
            real_time_aggregator: Service for real-time feature aggregation
            feast_service: Service for feature store interactions
        """
        self.features_config = features_config
        self.real_time_aggregator = real_time_aggregator
        self.feast_service = feast_service

        # State management per symbol
        self._symbol_contexts: Dict[str, Dict] = {}
        self._required_lookback = real_time_aggregator.get_required_lookback_periods()

    def initialize_for_symbol(
        self, symbol: str, timeframe: str, historical_data: DataFrame
    ) -> bool:
        """
        Initialize preprocessing for a specific symbol.

        This method sets up the necessary context and caches for
        real-time feature computation.
        """
        logger.info(f"Initializing real-time preprocessing for {symbol} {timeframe}")

        try:
            # Validate historical data
            if historical_data.empty:
                logger.error(f"No historical data provided for {symbol}")
                return False

            if len(historical_data) < self._required_lookback:
                logger.warning(
                    f"Limited historical data for {symbol}: {len(historical_data)} "
                    f"records (recommended: {self._required_lookback})"
                )

            # Ensure proper data formatting
            processed_data = self._prepare_historical_data(historical_data)

            # Create asset dataset for warm-up
            asset_data = AssetPriceDataSet(
                timeframe=timeframe,
                base_dataset=True,
                asset_price_dataset=processed_data,
            )

            # Warm up the feature aggregator cache
            self.real_time_aggregator.warm_up_cache(asset_data, symbol)

            # Store context for this symbol
            self._symbol_contexts[symbol] = {
                "timeframe": timeframe,
                "historical_data": processed_data.tail(self._required_lookback),
                "last_update": pd.Timestamp.now(),
                "initialized": True,
            }

            logger.info(f"Successfully initialized {symbol} for real-time processing")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {symbol}: {e}", exc_info=True)
            return False

    def _prepare_historical_data(self, historical_data: DataFrame) -> DataFrame:
        """Prepare and validate historical data format."""
        df = historical_data.copy()

        # Ensure required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure Time column is properly formatted
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time") if df.index.name != "Time" else df
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "Time"
        else:
            raise ValueError("Data must have a 'Time' column or DatetimeIndex")

        # Sort by time to ensure chronological order
        df = df.sort_index()

        return df

    def compute_features_for_latest_record(
        self, symbol: str, latest_record: Series
    ) -> Dict[str, float]:
        """
        Compute features for the latest price record.

        This method performs real-time feature computation using
        cached historical context and the new data point.
        """
        if symbol not in self._symbol_contexts:
            logger.error(f"Symbol {symbol} not initialized for real-time processing")
            return {}

        context = self._symbol_contexts[symbol]
        if not context.get("initialized", False):
            logger.error(f"Symbol {symbol} initialization incomplete")
            return {}

        try:
            # Get historical context
            historical_context = context["historical_data"]
            timeframe = context["timeframe"]

            # Ensure latest_record has proper format
            if "Time" not in latest_record.index and "Time" in latest_record:
                # Move Time from values to index if needed
                latest_record = latest_record.copy()
                time_val = latest_record.pop("Time")
                latest_record.name = pd.to_datetime(time_val)
            elif latest_record.name is None:
                latest_record.name = pd.Timestamp.now()

            # Compute features using real-time aggregator
            features = self.real_time_aggregator.compute_features_for_single_record(
                current_record=latest_record,
                historical_context=historical_context,
                symbol=symbol,
                timeframe=timeframe,
            )

            # Update historical context with latest record
            self._update_historical_context(symbol, latest_record)

            logger.debug(f"Computed {len(features)} features for {symbol}")
            return features

        except Exception as e:
            logger.error(f"Failed to compute features for {symbol}: {e}", exc_info=True)
            return {}

    def _update_historical_context(self, symbol: str, latest_record: Series) -> None:
        """Update the historical context with the latest record."""
        if symbol not in self._symbol_contexts:
            return

        context = self._symbol_contexts[symbol]
        historical_data = context["historical_data"]

        # Convert Series to DataFrame row
        new_row = pd.DataFrame([latest_record])
        new_row.index = [latest_record.name]

        # Append new row and maintain size limit
        updated_data = pd.concat([historical_data, new_row])
        updated_data = updated_data.tail(self._required_lookback)

        # Update context
        context["historical_data"] = updated_data
        context["last_update"] = pd.Timestamp.now()

    def get_online_features(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve features from the online feature store.

        This method leverages Feast's online serving capabilities
        to retrieve pre-computed features.
        """
        if not self.feast_service.is_enabled():
            logger.debug("Feast service not enabled for online features")
            return None

        if symbol not in self._symbol_contexts:
            logger.warning(
                f"Symbol {symbol} not initialized for online feature retrieval"
            )
            return None

        try:
            context = self._symbol_contexts[symbol]
            timeframe = context["timeframe"]

            # Create entity DataFrame for online feature retrieval
            _entity_df = pd.DataFrame(
                {
                    "Time": [timestamp],
                    "event_timestamp": [timestamp],
                    self.feast_service.config.entity_name: [
                        self.feast_service.get_entity_value(symbol, timeframe)
                    ],
                }
            )

            # Get feature references for all enabled features
            feature_refs = self._get_online_feature_references(timeframe)

            if not feature_refs:
                logger.debug("No online feature references available")
                return None

            # Retrieve online features using Feast
            logger.debug(f"Retrieving online features for {symbol}: {feature_refs}")

            # Note: This would use feast_service.feature_store.get_online_features()
            # but we need to implement the method in FeastService

            # For now, return None - this will be implemented when we extend FeastService
            logger.info(f"Online feature retrieval not yet implemented for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve online features for {symbol}: {e}")
            return None

    def _get_online_feature_references(self, timeframe: str) -> list:
        """Get list of feature references for online retrieval."""
        feature_refs = []

        for feature_def in self.features_config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                feature_view_name = self.feast_service.get_feature_view_name(
                    feature_def.name, param_set.param_hash, timeframe
                )

                # Get feature class to determine sub-features
                feature_class = (
                    self.real_time_aggregator.class_registry.feature_class_map.get(
                        feature_def.name
                    )
                )
                if feature_class:
                    # Create temporary instance to get sub-feature names
                    temp_instance = feature_class(
                        df_source=pd.DataFrame(), config=param_set
                    )
                    sub_features = temp_instance.get_sub_features_names()

                    for sub_feature in sub_features:
                        feature_refs.append(f"{feature_view_name}:{sub_feature}")

        return feature_refs

    def get_symbol_status(self, symbol: str) -> Dict:
        """Get status information for a symbol."""
        if symbol not in self._symbol_contexts:
            return {"initialized": False, "error": "Symbol not found"}

        context = self._symbol_contexts[symbol]
        return {
            "initialized": context.get("initialized", False),
            "timeframe": context.get("timeframe"),
            "historical_records": len(context.get("historical_data", [])),
            "last_update": context.get("last_update"),
            "cache_state": self.real_time_aggregator.get_cached_feature_state(),
        }

    def cleanup_symbol(self, symbol: str) -> None:
        """Clean up resources for a symbol."""
        if symbol in self._symbol_contexts:
            del self._symbol_contexts[symbol]
            logger.info(f"Cleaned up resources for {symbol}")

    def get_all_symbols(self) -> list:
        """Get list of all initialized symbols."""
        return list(self._symbol_contexts.keys())
