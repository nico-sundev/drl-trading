import math
from typing import Tuple

import pandas as pd

from ai_trading.model.split_dataset_container import SplitDataSetContainer


class SplitService:
    def __init__(self) -> None:
        pass

    def split_dataset(
        self, df: pd.DataFrame, split_ratios: Tuple[float, float, float]
    ) -> SplitDataSetContainer:
        """
        Split a dataset into training, validation and test sets.

        Args:
            df: The DataFrame to split
            split_ratios: Tuple of (train_ratio, val_ratio, test_ratio) that should sum to 1.0

        Returns:
            Container with train, validation and test datasets
        """
        train_ratio, val_ratio, test_ratio = split_ratios
        assert math.isclose(
            train_ratio + val_ratio + test_ratio,
            1.0,
        ), "Ratios must sum to 1.0"

        n = len(df)
        train_end = round(train_ratio * n)
        val_end = train_end + round(val_ratio * n)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        return SplitDataSetContainer(df_train, df_val, df_test)
