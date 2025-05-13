import math

import pandas as pd

from ai_trading.common.config.rl_model_config import RlModelConfig
from ai_trading.common.model.split_dataset_container import SplitDataSetContainer


class SplitService:

    def __init__(self, config: RlModelConfig):
        self.config = config

    # define config:
    # ticker
    # file path of datasets -> timeframes
    # train val tst ratio

    def split_dataset(self, df: pd.DataFrame) -> SplitDataSetContainer:
        assert math.isclose(
            self.config.training_split_ratio
            + self.config.validating_split_ratio
            + self.config.testing_split_ratio,
            1.0,
        ), "Ratios must sum to 1.0"

        n = len(df)
        train_end = round(self.config.training_split_ratio * n)
        val_end = train_end + round(self.config.validating_split_ratio * n)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        return SplitDataSetContainer(df_train, df_val, df_test)
