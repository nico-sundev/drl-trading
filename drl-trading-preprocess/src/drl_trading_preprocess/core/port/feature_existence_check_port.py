"""
Port for checking feature existence in the feature store.

Provides interface for querying existing features to prevent
redundant computation and support incremental processing.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.feature_computation_response import FeatureExistenceCheckResult


class IFeatureExistenceCheckPort(ABC):
    """
    Port for checking which features already exist in the feature store.

    Critical for preventing redundant computation and supporting
    incremental processing scenarios.
    """

    @abstractmethod
    def check_feature_existence(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_names: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> FeatureExistenceCheckResult:
        """
        Check which features exist for the given symbol and time range.

        Args:
            symbol: Trading symbol to check
            timeframe: Timeframe for the features
            feature_names: List of feature names to check
            start_time: Start of time range to check
            end_time: End of time range to check

        Returns:
            FeatureExistenceCheckResult with detailed existence information
        """
        pass

    @abstractmethod
    def get_latest_feature_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_name: str,
    ) -> Optional[datetime]:
        """
        Get the latest timestamp for a specific feature.

        Args:
            symbol: Trading symbol
            timeframe: Feature timeframe
            feature_name: Name of the feature

        Returns:
            Latest timestamp if feature exists, None otherwise
        """
        pass

    @abstractmethod
    def check_temporal_coverage(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_names: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, List[Dict[str, datetime]]]:
        """
        Check for temporal gaps in feature coverage.

        Args:
            symbol: Trading symbol
            timeframe: Feature timeframe
            feature_names: List of feature names to check
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping feature names to list of gap periods
        """
        pass

    @abstractmethod
    def get_feature_schema_version(
        self,
        symbol: str,
        timeframe: Timeframe,
    ) -> Optional[str]:
        """
        Get the schema version of existing features.

        Args:
            symbol: Trading symbol
            timeframe: Feature timeframe

        Returns:
            Schema version string if exists, None otherwise
        """
        pass

    @abstractmethod
    def validate_feature_compatibility(
        self,
        symbol: str,
        timeframe: Timeframe,
        feature_definitions: List[FeatureDefinition],
    ) -> Dict[str, bool]:
        """
        Validate if new feature definitions are compatible with existing ones.

        Args:
            symbol: Trading symbol
            timeframe: Feature timeframe
            feature_definitions: New feature definitions to validate

        Returns:
            Dictionary mapping feature names to compatibility status
        """
        pass
