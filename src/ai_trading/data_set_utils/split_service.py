import math
import pandas as pd
from ai_trading.config.rl_model_config import RlModelConfig
from ai_trading.model.split_dataset_container import SplitDataSetContainer

class SplitService:
    
    def __init__(self, rl_model_config: RlModelConfig):
        self.rl_model_config = rl_model_config
    
    # define config:
    # ticker
    # file path of datasets -> timeframes
    # train val tst ratio
    
    def split_dataset(self, df: pd.DataFrame) -> SplitDataSetContainer:
        assert math.isclose(self.rl_model_config.train_ratio + self.rl_model_config.val_ratio + self.rl_model_config.test_ratio, 1.0), "Ratios must sum to 1.0"
        
        n = len(df)
        train_end = round(self.rl_model_config.train_ratio * n)
        val_end = train_end + round(self.rl_model_config.val_ratio * n)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        return SplitDataSetContainer(df_train, df_val, df_test)