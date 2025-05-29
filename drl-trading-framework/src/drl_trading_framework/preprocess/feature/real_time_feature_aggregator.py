"""Real-time feature aggregator for inference support."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
from drl_trading_common.config.feature_config import (
    BaseParameterSetConfig,
    FeatureDefinition,
    FeaturesConfig,
)
from injector import inject
from pandas import DataFrame, Series

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.preprocess.feast.feast_service import FeastServiceInterface
from drl_trading_framework.preprocess.feature.feature_factory import (
    FeatureFactoryInterface,
)

logger = logging.getLogger(__name__)


class RealTimeFeatureAggregatorInterface(ABC):
    """
    Interface for real-time feature computation during inference.

    This interface supports single-record feature computation for real-time
    inference scenarios, complementing the batch FeatureAggregator.
    """

    @abstractmethod
    def compute_features_for_single_record(
        self,
        current_record: Series,
        historical_context: DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, float]:
        """
        Compute features for a single record in real-time.

        Args:
            current_record: Latest price data point as pandas Series
            historical_context: Recent historical data for lookback calculations
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Dict mapping feature names to computed values
        """
        pass

    @abstractmethod
    def get_required_lookback_periods(self) -> int:
        """
        Get the maximum lookback period required by any feature.

        Returns:
            Number of historical periods needed for feature computation
        """
        pass

    @abstractmethod
    def warm_up_cache(self, asset_data: AssetPriceDataSet, symbol: str) -> None:
        """
        Pre-populate feature cache for faster real-time computation.

        Args:
            asset_data: Historical data for cache warming
            symbol: Trading symbol
        """
        pass


class RealTimeFeatureAggregator(RealTimeFeatureAggregatorInterface):
    """
    Real-time feature aggregator for single-record inference.

    This class computes features incrementally for real-time inference,
    leveraging cached historical features and computing only what's needed
    for the latest data point.
    """

    @inject
    def __init__(
        self,
        config: FeaturesConfig,
        class_registry: FeatureFactoryInterface,
        feast_service: FeastServiceInterface,
    ) -> None:
        """
        Initialize the real-time feature aggregator.

        Args:
            config: Feature configuration
            class_registry: Registry of feature classes
            feast_service: Service for feature store interactions
        """
        self.config = config
        self.class_registry = class_registry
        self.feast_service = feast_service

        # Cache for storing intermediate computation state
        self._feature_cache: Dict[str, Dict] = {}

        # Pre-compute maximum lookback requirement
        self._max_lookback = self._calculate_max_lookback()

    def _calculate_max_lookback(self) -> int:
        """Calculate the maximum lookback period needed by any feature."""
        max_lookback = 50  # Default minimum for technical indicators

        # Analyze feature configurations to determine actual requirements
        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            # For each parameter set, check if it specifies lookback requirements
            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                # Extract common lookback parameters
                if hasattr(param_set, "period") and param_set.period:
                    max_lookback = max(max_lookback, param_set.period * 2)
                elif hasattr(param_set, "window") and param_set.window:
                    max_lookback = max(max_lookback, param_set.window * 2)

        return max_lookback

    def get_required_lookback_periods(self) -> int:
        """Get the maximum lookback period required by any feature."""
        return self._max_lookback

    def warm_up_cache(self, asset_data: AssetPriceDataSet, symbol: str) -> None:
        """
        Pre-populate feature cache using historical data.

        This method computes features for recent historical data to establish
        a baseline for incremental real-time computation.
        """
        logger.info(f"Warming up feature cache for {symbol} {asset_data.timeframe}")

        df = asset_data.asset_price_dataset
        if df.empty or len(df) < self._max_lookback:
            logger.warning(f"Insufficient data for cache warm-up: {len(df)} records")
            return

        # Use the last portion of historical data for warm-up
        warmup_data = df.tail(self._max_lookback).copy()

        # Pre-compute features using batch method for warm-up
        self._precompute_features_for_warmup(warmup_data, symbol, asset_data.timeframe)

    def _precompute_features_for_warmup(
        self, warmup_data: DataFrame, symbol: str, timeframe: str
    ) -> None:
        """Pre-compute features for cache warm-up using batch computation."""
        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                try:
                    # Create feature instance
                    feature_class = self.class_registry.feature_class_map.get(
                        feature_def.name
                    )
                    if not feature_class:
                        continue

                    feature_instance = feature_class(
                        df_source=warmup_data, config=param_set
                    )

                    # Compute feature for warm-up
                    computed_df = feature_instance.compute()
                    if computed_df is not None and not computed_df.empty:
                        cache_key = f"{feature_def.name}_{param_set.param_string}"
                        self._feature_cache[cache_key] = {
                            "last_values": computed_df.tail(5).to_dict("records"),
                            "feature_instance": feature_instance,
                            "sub_features": feature_instance.get_sub_features_names(),
                        }

                        logger.debug(f"Cached feature {cache_key} for warm-up")
                except Exception as e:
                    logger.warning(f"Failed to warm-up feature {feature_def.name}: {e}")

    def compute_features_for_single_record(
        self,
        current_record: Series,
        historical_context: DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, float]:
        """
        Compute features for a single record in real-time.

        This method leverages cached state and incremental computation
        to efficiently calculate features for the latest data point.
        """
        # Handle None inputs gracefully
        if current_record is None or historical_context is None:
            logger.warning("Received None input for feature computation")
            return {}

        logger.debug(
            f"Computing real-time features for {symbol} at {current_record.get('Time', 'unknown')}"
        )

        features = {}

        # Ensure we have sufficient historical context
        if historical_context.empty or len(historical_context) < 10:
            logger.warning("Insufficient historical context for feature computation")
            return features

        # Combine historical context with current record for computation
        extended_df = self._prepare_extended_dataframe(
            historical_context, current_record
        )

        for feature_def in self.config.feature_definitions:
            if not feature_def.enabled:
                continue

            for param_set in feature_def.parsed_parameter_sets:
                if not param_set.enabled:
                    continue

                computed_features = self._compute_single_feature_realtime(
                    feature_def, param_set, extended_df, symbol
                )

                if computed_features:
                    features.update(computed_features)

        return features

    def _prepare_extended_dataframe(
        self, historical_context: DataFrame, current_record: Series
    ) -> DataFrame:
        """Prepare DataFrame with historical context + current record."""
        # Convert current record to DataFrame row
        current_df = pd.DataFrame([current_record])

        # Ensure both have the same columns
        for col in historical_context.columns:
            if col not in current_df.columns:
                current_df[col] = None

        # Combine and ensure proper indexing
        extended_df = pd.concat([historical_context, current_df], ignore_index=False)

        # Ensure Time column exists and is properly formatted
        if "Time" in extended_df.columns and extended_df["Time"].dtype == "object":
            extended_df["Time"] = pd.to_datetime(extended_df["Time"])

        return extended_df

    def _compute_single_feature_realtime(
        self,
        feature_def: FeatureDefinition,
        param_set: BaseParameterSetConfig,
        extended_df: DataFrame,
        symbol: str,
    ) -> Optional[Dict[str, float]]:
        """Compute a single feature in real-time mode."""
        cache_key = f"{feature_def.name}_{param_set.param_string}"

        try:
            # Get or create feature instance
            if cache_key in self._feature_cache:
                cached_data = self._feature_cache[cache_key]
                feature_instance = cached_data.get("feature_instance")
                sub_features = cached_data.get("sub_features", [])
            else:
                # Create new feature instance
                feature_class = self.class_registry.feature_class_map.get(
                    feature_def.name
                )
                if not feature_class:
                    return None

                feature_instance = feature_class(
                    df_source=extended_df, config=param_set
                )
                sub_features = feature_instance.get_sub_features_names()

                # Initialize cache
                self._feature_cache[cache_key] = {
                    "feature_instance": feature_instance,
                    "sub_features": sub_features,
                    "last_values": [],
                }

            # Update feature instance with new data
            feature_instance.df_source = extended_df

            # Compute feature
            computed_df = feature_instance.compute()
            if computed_df is None or computed_df.empty:
                return None

            # Extract the latest computed values
            latest_row = computed_df.iloc[-1]
            result = {}

            for sub_feature in sub_features:
                if sub_feature in latest_row:
                    feature_name = (
                        f"{sub_feature}"  # Use same naming convention as batch
                    )
                    result[feature_name] = float(latest_row[sub_feature])

            # Update cache with latest values
            self._feature_cache[cache_key]["last_values"] = computed_df.tail(5).to_dict(
                "records"
            )

            return result

        except Exception as e:
            logger.warning(
                f"Failed to compute real-time feature {feature_def.name}: {e}"
            )
            return None

    def get_cached_feature_state(self) -> Dict:
        """Get current cache state for debugging/monitoring."""
        return {
            key: {
                "sub_features": data.get("sub_features", []),
                "last_values_count": len(data.get("last_values", [])),
            }
            for key, data in self._feature_cache.items()
        }
