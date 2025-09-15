"""Service for splitting datasets into training, validation, and testing portions."""

import logging
import math
from typing import Optional, Tuple, cast

import pandas as pd
from drl_trading_common.config.rl_model_config import RlModelConfig
from injector import inject

from drl_trading_core.common.model.split_dataset_container import (
    SplitDataSetContainer,
)

logger = logging.getLogger(__name__)


class SplitService:
    """
    Service for splitting a dataset into training, validation, and testing sets.

    This implementation maintains the chronological order of time series data,
    which is critical for financial datasets where sequence matters.
    """

    @inject
    def __init__(self, config: Optional[RlModelConfig] = None):
        """
        Initialize the split service.

        Args:
            config: Optional configuration containing default split ratios
        """
        self.config = config

    def split_dataset(
        self,
        df: pd.DataFrame,
        training_ratio: Optional[float] = None,
        validation_ratio: Optional[float] = None,
        testing_ratio: Optional[float] = None,
    ) -> SplitDataSetContainer:
        """
        Split a dataset into training, validation, and testing portions.

        If ratios are not provided, they will be taken from the config.
        All splits are performed in chronological order, maintaining the
        temporal sequence of the data, which is crucial for time series analysis.

        Args:
            df: The DataFrame to split
            training_ratio: Ratio for training data (overrides config if provided)
            validation_ratio: Ratio for validation data (overrides config if provided)
            testing_ratio: Ratio for testing data (overrides config if provided)

        Returns:
            A container with the split DataFrames

        Raises:
            ValueError: If the ratios don't sum to 1.0, are invalid, or no ratios are provided
            TypeError: If the input DataFrame is not valid
        """
        if df is None:
            raise TypeError("Input DataFrame cannot be None")

        if df.empty:
            logger.warning("Input DataFrame is empty, returning empty split container")
            empty_df = pd.DataFrame()
            return SplitDataSetContainer(empty_df, empty_df, empty_df)

        # Get ratios from parameters or config
        try:
            train_ratio = (
                training_ratio
                if training_ratio is not None
                else (
                    self.config.training_split_ratio
                    if self.config is not None
                    else None
                )
            )
            val_ratio = (
                validation_ratio
                if validation_ratio is not None
                else (
                    self.config.validating_split_ratio
                    if self.config is not None
                    else None
                )
            )
            test_ratio = (
                testing_ratio
                if testing_ratio is not None
                else (
                    self.config.testing_split_ratio if self.config is not None else None
                )
            )
        except AttributeError as e:
            logger.error(f"Failed to access configuration attributes: {e}")
            raise ValueError("Invalid configuration object provided") from e

        # Validate ratios
        if any(ratio is None for ratio in [train_ratio, val_ratio, test_ratio]):
            logger.error(
                "Missing split ratios - must be provided in parameters or config"
            )
            raise ValueError(
                "Split ratios must be provided either in parameters or config"
            )

        # Validate ratio values
        self._validate_ratios(train_ratio, val_ratio, test_ratio)

        try:
            # Calculate split indices
            n = len(df)
            train_end = round(cast(float, train_ratio) * n)
            val_end = train_end + round(cast(float, val_ratio) * n)

            # Split the dataframe
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]

            logger.debug(
                f"Split DataFrame into training ({len(df_train)}), "
                f"validation ({len(df_val)}), test ({len(df_test)}) sets"
            )

            return SplitDataSetContainer(df_train, df_val, df_test)

        except Exception as e:
            logger.error(f"Error during dataset splitting: {e}")
            raise RuntimeError(f"Failed to split dataset: {e}") from e

    def _validate_ratios(
        self,
        training_ratio: Optional[float],
        validation_ratio: Optional[float],
        testing_ratio: Optional[float],
    ) -> None:
        """
        Validate that the provided split ratios are valid.

        Args:
            training_ratio: The training data ratio
            validation_ratio: The validation data ratio
            testing_ratio: The testing data ratio

        Raises:
            ValueError: If ratios are outside valid range or don't sum to 1.0
        """
        # Validate each ratio is between 0 and 1
        for name, ratio in [
            ("Training", training_ratio),
            ("Validation", validation_ratio),
            ("Testing", testing_ratio),
        ]:
            if ratio is None:
                continue

            if not isinstance(ratio, (int, float)):
                raise ValueError(
                    f"{name} ratio must be a number, got {type(ratio).__name__}"
                )

            if ratio < 0 or ratio > 1:
                raise ValueError(f"{name} ratio must be between 0 and 1, got {ratio}")

        # Cast to float for arithmetic operations
        train_ratio = cast(float, training_ratio)
        val_ratio = cast(float, validation_ratio)
        test_ratio = cast(float, testing_ratio)
        # Validate sum is close to 1.0
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-10):
            total = train_ratio + val_ratio + test_ratio
            logger.error(f"Split ratios sum to {total}, expected 1.0")
            raise ValueError(f"Ratios must sum to 1.0, got: {total}")

    def calculate_split_indices(
        self, length: int, ratios: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """
        Calculate split indices based on dataset length and ratios.

        Args:
            length: Total length of the dataset
            ratios: Tuple of (training_ratio, validation_ratio, testing_ratio)

        Returns:
            Tuple of (train_end_idx, validation_end_idx)

        Raises:
            ValueError: If length is negative or ratios are invalid
            TypeError: If inputs have incorrect types
        """
        if not isinstance(length, int):
            raise TypeError(f"Length must be an integer, got {type(length).__name__}")

        if length < 0:
            raise ValueError(f"Length cannot be negative, got {length}")

        if not isinstance(ratios, tuple) or len(ratios) != 3:
            raise TypeError("Ratios must be a tuple of 3 float values")

        try:
            train_ratio, val_ratio, test_ratio = ratios
            # Validate ratios
            self._validate_ratios(train_ratio, val_ratio, test_ratio)

            # Calculate indices
            train_end = round(train_ratio * length)
            val_end = train_end + round(val_ratio * length)

            logger.debug(
                f"Calculated split indices: train_end={train_end}, val_end={val_end}"
            )
            return train_end, val_end

        except Exception as e:
            logger.error(f"Error calculating split indices: {e}")
            raise ValueError(f"Failed to calculate split indices: {e}") from e
