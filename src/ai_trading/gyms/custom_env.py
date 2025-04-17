import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pandas import DataFrame
import logging
import random

from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.gyms.utils.trading_env_utils import TradingEnvUtils, TradingDirection

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    def __init__(
        self,
        env_data_source: DataFrame,
        env_config: EnvironmentConfig,
        feature_start_index: int,
    ):
        """Initialize the trading environment.

        Args:
            env_data_source: DataFrame containing price data and computed features
            env_config: Configuration for the trading environment
            feature_start_index: Index in env_data_source columns where computed/normalized features start
        """
        super(TradingEnv, self).__init__()
        self.env_config = env_config
        self.env_data_source = env_data_source
        self.feature_start_index = feature_start_index

        # Initialize from environment config
        self.initial_balance = env_config.start_balance
        self.balance = env_config.start_balance

        # Position tracking
        self.current_step = 0
        self.done = False
        self.position_state = 0  # 1 for long, -1 for short, 0 for no position
        self.time_in_position = 0
        self.position_open_price = None
        self.number_contracts_owned = 0.0
        self.pnl = 0.0
        self.current_leverage = 1.0
        self.liquidation_price = None
        self.was_liquidated = False
        self.last_liquidation_loss = 0.0

        # Discrete actions:
        # 0 (Open Long),
        # 1 (Close Position),
        # 2 (Open Short),
        # 3 (Hold Position),
        # 4 (Partial Close),
        # 5 (Await Entry)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(6),  # Discrete actions
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
                ),  # Continuous space for partial close
                spaces.Box(
                    low=1,
                    high=20,
                    shape=(1,),
                    dtype=np.int16,
                ),  # Continuous space for leverage
            )
        )

        # Observation space calculation based on computed features only
        feature_columns = self.env_data_source.columns[feature_start_index:]
        num_features = len(feature_columns)
        low = np.full(num_features, -np.inf)
        high = np.full(num_features, np.inf)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        """Reset environment state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_state = 0
        self.number_contracts_owned = 0.0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0.0
        self.done = False
        self.current_leverage = 1.0
        self.liquidation_price = None
        self.was_liquidated = False
        self.last_liquidation_loss = 0.0
        return self._next_observation()

    def _next_observation(self):
        """Get the next observation from the environment.

        Returns only the computed and normalized features starting from feature_start_index,
        excluding strategy-specific data like prices, ATR, etc.

        Returns:
            numpy.ndarray: Array of computed features in float32 format
        """
        feature_set = self.env_data_source.iloc[self.current_step]
        # Select only computed features starting from feature_start_index
        computed_features = feature_set.iloc[self.feature_start_index :].values
        return computed_features.astype(np.float32)

    def _update_liquidation_price(self, leverage: float):
        """Update liquidation price for leveraged positions."""
        if leverage > 1.0 and self.position_state != 0:
            self.liquidation_price = TradingEnvUtils.calculate_liquidation_price(
                self.position_open_price, self.number_contracts_owned, leverage
            )
            self.current_leverage = leverage
        else:
            self.liquidation_price = None
            self.current_leverage = 1.0

    def _check_liquidation(self, high: float, low: float) -> bool:
        """Check if position should be liquidated based on high/low prices."""
        if self.liquidation_price is None or not TradingEnvUtils.has_active_position(
            self.position_state
        ):
            return False

        if self.position_state == 1:  # Long position
            return low <= self.liquidation_price
        else:  # Short position
            return high >= self.liquidation_price

    def _handle_liquidation(self, current_price: float):
        """Handle liquidation of a position."""
        logger.info(f"Position liquidated at price {current_price}")
        # Then close the position
        self._close_position(current_price)

    def _calculate_dynamic_slippage(
        self, direction: TradingDirection, current_price: float
    ) -> float:
        """Calculate dynamic slippage based on ATR and trade direction.

        This method calculates slippage as a percentage of the price (e.g., 0.01 = 1%) to match
        TradingEnvUtils' expectation of percentage-based slippage values. The slippage is:
        1. Based on ATR relative to current price to reflect market volatility
        2. Works against the trade with env_config.slippage_against_trade_probability (default 60%)
        3. Used both for position entry price adjustment and fee calculations
        """
        current_data = self.env_data_source.iloc[self.current_step]
        atr = current_data.atr

        # Base slippage as a percentage (ATR relative to price)
        base_slippage = (atr / current_price) * self.env_config.slippage_atr_based

        # Determine if slippage works against the trade
        against_trade = (
            random.random() < self.env_config.slippage_against_trade_probability
        )

        # For long positions: negative slippage means price goes up before entry
        # For short positions: positive slippage means price goes down before entry
        if against_trade:
            return (
                base_slippage if direction == TradingDirection.LONG else -base_slippage
            )
        else:
            return (
                -base_slippage if direction == TradingDirection.LONG else base_slippage
            )

    def _take_action(self, action):
        discrete_action = action[0]
        open_position_percentage = action[1][0]
        partial_close_percentage = action[2][0]
        leverage = action[3][0]

        feature_set = self.env_data_source.iloc[self.current_step]
        current_price = feature_set.price

        if discrete_action == 0:  # Open Long
            if self.position_state != 0:
                self._close_position(current_price)

            self.position_state = 1
            slippage = self._calculate_dynamic_slippage(
                TradingDirection.LONG, current_price
            )
            # Apply slippage to entry price
            self.position_open_price = current_price * (1 + slippage)
            # Calculate contracts based on original price to maintain correct position size
            self.number_contracts_owned = TradingEnvUtils.calculate_number_contracts(
                TradingDirection.LONG,
                open_position_percentage,
                current_price,
                self.balance,
                self.env_config.fee,
                0,  # Don't apply slippage twice
                leverage,
            )
            self.time_in_position = 0
            self.pnl = 0.0
            self._update_liquidation_price(leverage)

        elif discrete_action == 1:  # Close Position
            if self.position_state != 0:
                self._close_position(current_price)

        elif discrete_action == 2:  # Open Short
            if self.position_state != 0:
                self._close_position(current_price)

            self.position_state = -1
            slippage = self._calculate_dynamic_slippage(
                TradingDirection.SHORT, current_price
            )
            # Apply slippage to entry price
            self.position_open_price = current_price * (1 + slippage)
            # Calculate contracts based on original price to maintain correct position size
            self.number_contracts_owned = TradingEnvUtils.calculate_number_contracts(
                TradingDirection.SHORT,
                open_position_percentage,
                current_price,
                self.balance,
                self.env_config.fee,
                0,  # Don't apply slippage twice
                leverage,
            )
            self.time_in_position = 0
            self.pnl = 0.0
            self._update_liquidation_price(leverage)

        elif discrete_action == 3:  # Hold Position
            if self.position_state != 0:
                self._update_pnl(current_price)

        elif discrete_action == 4:  # Partial Close
            if self.position_state != 0 and 0 < partial_close_percentage <= 1:
                self._partial_close_position(current_price, partial_close_percentage)

    def _partial_close_position(self, current_price: float, percentage: float):
        self._update_pnl(current_price)
        closing_fee = TradingEnvUtils.calculate_close_fee(
            current_price,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(
                (
                    TradingDirection.LONG
                    if self.position_state == 1
                    else TradingDirection.SHORT
                ),
                current_price,
            ),
            percentage,
        )
        self.balance += (self.pnl * percentage) - closing_fee

        self.number_contracts_owned = self.number_contracts_owned * (1 - percentage)
        if self.number_contracts_owned == 0:
            self.position_state = 0
            self.time_in_position = 0
            self.position_open_price = None
            self.pnl = 0.0
            self.liquidation_price = None
            self.current_leverage = 1.0

    def _update_pnl(self, current_price: float):
        direction = TradingEnvUtils.get_position_direction(self.position_state)
        self.pnl = TradingEnvUtils.calculate_pnl(
            current_price,
            self.position_open_price,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(direction, current_price),
        )

    def _close_position(self, current_price: float):
        self._update_pnl(current_price)
        closing_fee = TradingEnvUtils.calculate_close_fee(
            current_price,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(
                (
                    TradingDirection.LONG
                    if self.position_state == 1
                    else TradingDirection.SHORT
                ),
                current_price,
            ),
        )
        self.balance += self.pnl - closing_fee
        self.position_state = 0
        self.number_contracts_owned = 0.0
        self.position_open_price = None
        self.current_leverage = 1.0

    def _calculate_reward(self) -> float:
        """Calculate reward for the current position state.

        The reward function considers:
        1. For profitable positions:
           - Reward increases with the square root of time
           - Reward scaled by in-money factor
        2. For losing positions:
           - Penalty increases more quickly with time (power of 1.5)
           - Penalty scaled by out-of-money factor
           - This naturally handles liquidation cases as the PnL already accounts
             for leveraged losses

        Returns:
            float: The calculated reward value
        """
        if self.pnl > 0:
            # For profitable positions, reward increases with time but at a decreasing rate
            return (
                self.env_config.in_money_factor
                * self.pnl
                * np.sqrt(self.time_in_position)
            )
        else:
            # For losing positions, penalty increases more quickly with time
            # This also handles liquidation as PnL already accounts for leveraged losses
            return (
                -self.env_config.out_of_money_factor
                * abs(self.pnl)
                * (self.time_in_position**1.5)
            )

    def step(self, action):
        current_data = self.env_data_source.iloc[self.current_step]

        if TradingEnvUtils.has_active_position(self.position_state):
            self.time_in_position += 1

        # Check for liquidation before taking action
        if self._check_liquidation(current_data.high, current_data.low):
            self._handle_liquidation(
                current_data.low if self.position_state == 1 else current_data.high
            )
        else:
            self._take_action(action)

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.env_data_source) - 1:
            self.done = True

        # Calculate reward
        reward = self._calculate_reward()

        # Reset after reward calculation
        if not TradingEnvUtils.has_active_position(self.position_state):
            self.pnl = 0.0
            self.time_in_position = 0
            self.liquidation_price = None

        return self._next_observation(), reward, self.done, {}

    def render(self, mode="human"):
        logger.info(f"Step: {self.current_step}")
        logger.info(f"Price Data: {self.env_data_source.iloc[self.current_step]}")
        logger.info(
            f'Position: {"Long" if self.position_state == 1 else "Short" if self.position_state == -1 else "None"}'
        )
        logger.info(f"Time in Position: {self.time_in_position}")
        logger.info(f"Profit: {self.pnl}")
        logger.info(f"Balance: {self.balance}")
        logger.info(f"Position Size: {self.number_contracts_owned}")
        if self.liquidation_price is not None:
            logger.info(f"Liquidation Price: {self.liquidation_price}")
            logger.info(f"Current Leverage: {self.current_leverage}x")
