import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pandas import DataFrame

from ai_trading.config.environment_config import EnvironmentConfig


from enum import Enum


class TradingDirection(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0


class TradingEnv(gym.Env):
    def __init__(self, env_data_source: DataFrame, env_config: EnvironmentConfig):
        super(TradingEnv, self).__init__()
        self.env_config = env_config
        self.env_data_source = env_data_source
        self.in_money_factor = 1.0  # Factor for in-the-money positions
        self.out_of_money_factor = 1.0  # Factor for out-of-the-money positions
        

        # Discrete actions:
        # 0 (Open Long),
        # 1 (Close Position),
        # 2 (Open Short),
        # 3 (Hold Position),
        # 4 (Partial Close),
        # 5 (Await Entry)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(6),  # 5 discrete actions
                spaces.Box(
                    low=env_config.min_percentage_open_position,
                    high=env_config.max_percentage_open_position,
                    shape=(1,),
                    dtype=np.float32,
                ),  # Continuous space for open positions
                spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),  # Continuous space for partial close (only used when action == 4)
                spaces.Box(
                    low=1,
                    high=20,
                    shape=(1,),
                    dtype=np.int16,
                ),  # Continuous space for leverage
            )
        )

        num_of_features = len(
            self.env_data_source.columns
        )  # Includes computed features and context columns
        computational_excluded_features = 1  # close price
        total_features = num_of_features - computational_excluded_features

        # Define bounds for the observation space
        low = np.full(total_features, -np.inf)  # Replace with specific bounds if known
        high = np.full(total_features, np.inf)  # Replace with specific bounds if known

        # Define the observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Price data (for simplicity)
        self.env_data_source = env_data_source
        self.current_step = 0
        self.done = False

        # Trading related attributes
        self.initial_balance = env_config.start_balance
        self.balance = env_config.start_balance
        self.position_state = 0  # 1 for long, -1 for short, 0 for no position
        self.time_in_position = 0
        self.position_open_price = None
        self.number_contracts_owned = (
            0  # New variable to track position size (fraction of the full position)
        )
        self.pnl = 0

        # TODO: Implement slippage based on ATR
        self.slippage = 0.0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_state = 0
        self.number_contracts_owned = 0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        feature_set = self.env_data_source.iloc[self.current_step]
        return np.array(
            [
                feature_set,
                self.position_state,
                self.time_in_position,
                self.pnl,
                self.number_contracts_owned,
            ],
            dtype=object  # Use dtype=object to handle mixed types
        )

    def _calculate_number_contracts(
        self,
        direction: TradingDirection,
        open_position_percentage: float,
        current_price: float,
        leverage: int = 1.0,
    ) -> float:
        """Calculate the number of contracts to open based on the open position percentage and current price.

        Args:
            open_position_percentage (_type_): Percentage of the balance to open a new position.
            current_price (_type_): Current price of the asset.

        Returns:
            float: Number of contracts to open.
        """
        total_position_value = (
            self.balance * open_position_percentage / 100.0
        ) * leverage
        total_position_value_excluding_fee = total_position_value * (
            1.0 - self.env_config.fee - self.slippage
        )
        return (total_position_value_excluding_fee / current_price) * direction.value

    def _calculate_liquidation_price(
        self, position_open_price: float, leverage: int = 1.0
    ) -> float:
        """
        Calculate the liquidation price for a position based on the open price, leverage, and trade direction.

        Args:
            position_open_price (float): The price at which the position was opened.
            leverage (int): The leverage used for the position.

        Returns:
            float: The calculated liquidation price.
        """
        if leverage <= 1:
            raise ValueError(
                "Leverage must be greater than 1 to calculate liquidation price."
            )

        # Adjust the formula to account for non-linearity at higher leverage
        adjustment_factor = 1 - (1 / (leverage**1.1))  # Exponential adjustment

        if self.number_contracts_owned > 0:  # Long trade
            liquidation_price = position_open_price * adjustment_factor
        elif self.number_contracts_owned < 0:  # Short trade
            liquidation_price = position_open_price / adjustment_factor
        else:
            raise ValueError("No active position to calculate liquidation price.")

        return liquidation_price

    def _take_action(self, action):
        discrete_action = action[0]  # Discrete action
        open_position_percentage = action[1][0]
        partial_close_percentage = action[2][0]

        feature_set = self.env_data_source[self.current_step]

        if discrete_action == 0:  # Open Long

            if self.position_state != 0:
                self._close_position(feature_set.price)

            self.position_state = 1
            self.position_open_price = feature_set.price
            self.number_contracts_owned = self._calculate_number_contracts(
                TradingDirection.LONG, open_position_percentage, feature_set.price
            )
            self.time_in_position = 0
            self.pnl = 0

        elif discrete_action == 1:  # Close Position
            if self.position_state != 0:
                self._close_position(feature_set.price)

        elif discrete_action == 2:  # Open Short

            if self.position_state != 0:
                # Trade Reversal
                self._close_position(feature_set.price)

            self.position_state = -1
            self.position_open_price = feature_set.price
            self.number_contracts_owned = self._calculate_number_contracts(
                TradingDirection.SHORT, open_position_percentage, feature_set.price
            )
            self.time_in_position = 0
            self.pnl = 0

        elif discrete_action == 3:  # Hold Position
            if self.position_state != 0:
                self.time_in_position += 1
                self._update_pnl(feature_set.price)

        elif discrete_action == 4:  # Partial Close
            if self.position_state != 0 and 0 < partial_close_percentage <= 1:
                self._partial_close_position(
                    feature_set.price, partial_close_percentage
                )

    def _partial_close_position(self, current_price, percentage):
        self._update_pnl(current_price)
        self.balance += (self.pnl * percentage) - self._calculate_close_fee(
            current_price, 1.0
        )

        self.number_contracts_owned = self.number_contracts_owned * (
            1 - percentage
        )  # Reduce position size by the closed amount
        if self.number_contracts_owned == 0:  # Fully closed
            self.position_state = 0
            self.time_in_position = 0
            self.position_open_price = None
            self.pnl = 0

    def _update_pnl(self, current_price):
        self.pnl = (
            current_price - self.position_open_price
        ) * self.number_contracts_owned - self._calculate_close_fee(current_price, 1.0)

    def _calculate_close_fee(self, current_price, percentage: float = 1.0):
        # Calculate the closing fee based on the current price and the number of contracts owned
        return abs(current_price * self.number_contracts_owned * percentage) * (
            self.env_config.fee + self.slippage
        )

    def _close_position(self, current_price):
        self._update_pnl(current_price)
        self.balance += self.pnl - self._calculate_close_fee(
            current_price, 1.0
        )  # Update balance based on profit/loss from the position
        self.position_state = 0
        self.number_contracts_owned = 0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0

    def step(self, action):
        self._take_action(action)

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.env_data_source) - 1:
            self.done = True

        # Reward based on profit or loss in the position
        reward = 0
        if self.position_state != 0:
            if self.pnl > 0:
                reward = self.in_money_factor * self.pnl * self.time_in_position
            elif self.pnl < 0:
                reward = (
                    self.out_of_money_factor * abs(self.pnl) * self.time_in_position
                )

        # Get the next observation
        observation = self._next_observation()

        return observation, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Price: {self.env_data_source[self.current_step]}")
        print(
            f'Position: {"Long" if self.position_state == 1 else "Short" if self.position_state == -1 else "None"}'
        )
        print(f"Time in Position: {self.time_in_position}")
        print(f"Profit: {self.pnl}")
        print(f"Balance: {self.balance}")
        print(f"Position Size: {self.number_contracts_owned}")


from enum import Enum


class TradingDirection(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0
