from typing import Any, Dict, List, Optional

import numpy as np
from drl_trading_core import BaseTradingEnv
from gymnasium import spaces
from pandas import DataFrame

from drl_trading_common.config.environment_config import EnvironmentConfig


class MyCustomTradingEnv(BaseTradingEnv):
    def __init__(
        self,
        env_data_source: DataFrame,
        env_config: EnvironmentConfig,
        context_columns: Optional[List[str]] = None,
    ):
        super().__init__(env_data_source, env_config, context_columns)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example: discrete actions (Buy, Sell, Hold)
        self.action_space = spaces.Discrete(3)

        # Example: observation space (continuous features)
        # The shape should match the number of features your environment will provide
        # For example, if you have 10 features in your observation:
        num_features = len(self.env_data_source.columns)  # Or a more specific count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

        self.current_step = 0
        self.initial_balance = env_config.start_balance
        self.balance = self.initial_balance
        self.done = False

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.done = False
        return self._next_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Implement your environment logic for taking a step
        # This is a placeholder implementation
        self.current_step += 1

        # Placeholder reward and done logic
        reward = 0  # Calculate reward based on action and market change
        # Example: if action leads to profit, reward is positive

        if self.current_step >= len(self.env_data_source) - 1:
            self.done = True
            terminated = True
        else:
            terminated = False

        truncated = self.done  # can also be based on other conditions

        # Update balance based on action (simplified)
        if action == 0:  # Buy
            pass  # Implement buy logic
        elif action == 1:  # Sell
            pass  # Implement sell logic
        # else Hold, no change to balance from action itself

        # Check for conditions like max drawdown if necessary
        # if self.balance < self.initial_balance * (1 - self.env_config.max_alltime_drawdown):
        #     self.done = True

        return self._next_observation(), reward, terminated, truncated, self._get_info()

    def _next_observation(self) -> np.ndarray:
        # Return the observation for the current step
        # This should be a numpy array matching self.observation_space
        obs = self.env_data_source.iloc[self.current_step].values.astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        # Return auxiliary information, e.g., current balance, trades, etc.
        return {"balance": self.balance, "current_step": self.current_step}

    def render(self, mode="human"):
        # Implement rendering logic if needed (e.g., for visualization)
        print(f"Step: {self.current_step}, Balance: {self.balance}")

    def close(self):
        # Implement any cleanup logic
        pass
