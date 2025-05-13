"""Service for managing context-related features required by the trading environment."""

import logging
from typing import List, Optional

import pandas_ta as ta
from pandas import DataFrame

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.trading_constants import (
    ALL_CONTEXT_COLUMNS,
    DERIVED_CONTEXT_COLUMNS,
    PRIMARY_CONTEXT_COLUMNS,
)
from drl_trading_framework.preprocess.data_set_utils.util import ensure_datetime_index

logger = logging.getLogger(__name__)


class ContextFeatureService:
    """
    Service responsible for managing context-related features required by the trading environment.

    This service centralizes the logic for preparing, validating, and identifying context features
    that are required for the functioning of the trading environment but are not part of the
    feature set used for machine learning.

    Context columns are divided into two categories:
    1. Primary columns: Must exist in the raw data (e.g., OHLC)
    2. Derived columns: Computed from primary columns (e.g., ATR)
    """

    def __init__(self, atr_period: int = 14):
        """
        Initialize the ContextFeatureService.

        Args:
            atr_period: Period to use for ATR calculation (default: 14)
        """
        self.atr_period = atr_period

    def prepare_context_features(self, base_dataset: AssetPriceDataSet) -> DataFrame:
        """
        Prepares a DataFrame with only the essential context-related features needed by the trading environment.

        This method:
        1. Validates that all required primary columns exist in the dataset
        2. Computes any necessary derived columns (e.g., ATR)
        3. Returns a DataFrame containing only the context columns

        Args:
            base_dataset: The base dataset containing OHLC data

        Returns:
            DataFrame: A DataFrame containing only the columns needed for the trading environment context

        Raises:
            ValueError: If essential primary columns are missing from the base dataset
        """
        # Get the DataFrame from the base dataset
        df = base_dataset.asset_price_dataset.copy()
        # Step 1: Validate primary columns exist
        self._validate_primary_columns(df)

        # Step 2: Start with only required primary columns
        # (excluding optional ones like Open and Volume)
        selected_columns = [col for col in PRIMARY_CONTEXT_COLUMNS if col in df.columns]

        # Step 3: Add/compute derived columns
        df = self._compute_derived_columns(df)

        # Add derived columns to selected columns
        for derived_col in DERIVED_CONTEXT_COLUMNS:
            selected_columns.append(derived_col)

        # Return only the selected columns
        result_df = df[selected_columns]
        logger.info(
            f"Prepared context features with columns: {', '.join(selected_columns)}"
        )
        return result_df

    def _validate_primary_columns(self, df: DataFrame) -> None:
        """
        Validates that all required primary columns exist in the DataFrame.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If any required primary columns are missing
        """
        missing_columns = [
            col for col in PRIMARY_CONTEXT_COLUMNS if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Essential primary columns missing from dataset: {missing_columns}"
            )

    def _compute_derived_columns(self, df: DataFrame) -> DataFrame:
        """
        Computes derived context columns if they don't already exist.

        Args:
            df: DataFrame containing primary columns

        Returns:
            DataFrame: Input DataFrame with derived columns added
        """
        # Compute ATR
        logger.debug(f"Computing ATR with period {self.atr_period}")
        # Preserve existing ATR if present
        if "Atr" not in df.columns or df["Atr"].isna().all():
            df["Atr"] = ta.atr(
                df["High"], df["Low"], df["Close"], timeperiod=self.atr_period
            )
        return df

    def is_context_column(self, column_name: str) -> bool:
        """
        Determines if a column is a context column.

        Args:
            column_name: The name of the column to check

        Returns:
            bool: True if the column is a context column, False otherwise
        """
        return column_name in ALL_CONTEXT_COLUMNS

    def is_primary_column(self, column_name: str) -> bool:
        """
        Determines if a column is a primary context column.

        Primary columns must exist in the raw data, unlike derived columns
        which are computed.

        Args:
            column_name: The name of the column to check

        Returns:
            bool: True if the column is a primary context column, False otherwise
        """
        return column_name in PRIMARY_CONTEXT_COLUMNS

    def is_derived_column(self, column_name: str) -> bool:
        """
        Determines if a column is a derived context column.

        Derived columns are computed from primary columns and don't
        need to exist in the raw data.

        Args:
            column_name: The name of the column to check

        Returns:
            bool: True if the column is a derived context column, False otherwise
        """
        return column_name in DERIVED_CONTEXT_COLUMNS

    def get_context_columns(self, df: Optional[DataFrame] = None) -> List[str]:
        """
        Returns a list of context column names.

        If a DataFrame is provided, returns only the context columns present in the DataFrame.
        Otherwise, returns all possible context columns.

        Args:
            df: Optional DataFrame to check for context columns

        Returns:
            List[str]: List of context column names
        """
        if df is not None:
            return [col for col in ALL_CONTEXT_COLUMNS if col in df.columns]
        return list(ALL_CONTEXT_COLUMNS)

    def get_feature_columns(self, df: DataFrame) -> List[str]:
        """
        Returns a list of feature column names (non-context columns) in the DataFrame.

        Args:
            df: DataFrame to extract feature columns from

        Returns:
            List[str]: List of feature column names
        """
        return [col for col in df.columns if not self.is_context_column(col)]

    def merge_context_features(
        self, computed_dataframe: DataFrame, context_features: DataFrame
    ) -> DataFrame:
        """
        Merges context-related features into a DataFrame with computed features.

        Args:
            computed_dataframe: DataFrame containing computed features with DatetimeIndex
            context_features: DataFrame containing context-related features with DatetimeIndex

        Returns:
            DataFrame: The merged DataFrame with context-related features
        """

        # Ensure both DataFrames have DateTimeIndex
        computed_dataframe = ensure_datetime_index(
            computed_dataframe, "computed dataframe"
        )
        context_features = ensure_datetime_index(context_features, "context features")

        # Join on DatetimeIndex for better performance
        return computed_dataframe.join(context_features, how="left")
