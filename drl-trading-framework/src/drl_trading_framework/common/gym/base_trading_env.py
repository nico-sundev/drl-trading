from typing import List, Optional

import gymnasium as gym
from drl_trading_common.config.environment_config import EnvironmentConfig
from pandas import DataFrame


class BaseTradingEnv(gym.Env):

    def __init__(
        self,
        env_data_source: DataFrame,
        env_config: EnvironmentConfig,
        context_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the trading environment.

        Args:
            env_data_source (DataFrame): Data source for the environment.
            env_config (EnvironmentConfig): Configuration settings for the environment.
            context_columns (Optional[List[str]]): List of context columns to be used in the environment.
        """
        super().__init__()
        self.env_data_source = env_data_source
        self.env_config = env_config
        self.context_columns = context_columns if context_columns else []
