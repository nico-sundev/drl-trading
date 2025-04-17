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
    def __init__(self, env_data_source: DataFrame, env_config: EnvironmentConfig):
        super(TradingEnv, self).__init__()
        self.env_config = env_config
        self.env_data_source = env_data_source

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

        # Observation space
        num_of_features = len(self.env_data_source.columns)
        computational_excluded_features = 1  # close price
        total_features = num_of_features - computational_excluded_features
        low = np.full(total_features, -np.inf)
        high = np.full(total_features, np.inf)
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
        feature_set = self.env_data_source.iloc[self.current_step]
        return np.array(
            [
                feature_set,
                self.position_state,
                self.time_in_position,
                self.pnl,
                self.number_contracts_owned,
                self.current_leverage,
                self.liquidation_price if self.liquidation_price is not None else 0.0,
            ],
            dtype=object
        )

    def _update_liquidation_price(self, leverage: float):
        """Update liquidation price for leveraged positions."""
        if leverage > 1.0 and self.position_state != 0:
            self.liquidation_price = TradingEnvUtils.calculate_liquidation_price(
                self.position_open_price,
                self.number_contracts_owned,
                leverage
            )
            self.current_leverage = leverage
        else:
            self.liquidation_price = None
            self.current_leverage = 1.0

    def _check_liquidation(self, high: float, low: float) -> bool:
        """Check if position should be liquidated based on high/low prices."""
        if self.liquidation_price is None or self.position_state == 0:
            return False

        if self.position_state == 1:  # Long position
            return low <= self.liquidation_price
        else:  # Short position
            return high >= self.liquidation_price

    def _handle_liquidation(self, current_price: float):
        """Handle liquidation of a position."""
        logger.info(f"Position liquidated at price {current_price}")
        # Update PnL first to get the correct loss amount
        self._update_pnl(current_price)
        # Store the loss for reward calculation
        liquidation_loss = self.pnl
        # Then close the position
        self._close_position(current_price)
        # Set liquidation flag and store the loss
        self.was_liquidated = True
        self.last_liquidation_loss = liquidation_loss

    def _calculate_dynamic_slippage(self, direction: TradingDirection, current_price: float) -> float:
        """Calculate dynamic slippage based on ATR and trade direction.
        
        This method calculates slippage as a percentage of the price (e.g., 0.01 = 1%) to match
        TradingEnvUtils' expectation of percentage-based slippage values. The slippage is:
        1. Based on ATR relative to current price to reflect market volatility
        2. Works against the trade with env_config.slippage_against_trade_probability (default 60%)
        3. Used both for position entry price adjustment and fee calculations
        
        Args:
            direction: The direction of the trade (LONG/SHORT)
            current_price: The current price
            
        Returns:
            float: The calculated slippage as a percentage (0.01 = 1%)
        """
        # Get ATR value from the current data
        current_data = self.env_data_source.iloc[self.current_step]
        atr = current_data.atr
        
        # Base slippage as a percentage (ATR relative to price)
        base_slippage = (atr / current_price) * self.env_config.slippage_atr_based
        
        # Determine if slippage works against the trade (60% probability by default)
        against_trade = random.random() < self.env_config.slippage_against_trade_probability
        
        # For long positions: negative slippage means price goes up before entry
        # For short positions: positive slippage means price goes down before entry
        if against_trade:
            return base_slippage if direction == TradingDirection.LONG else -base_slippage
        else:
            return -base_slippage if direction == TradingDirection.LONG else base_slippage

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
            slippage = self._calculate_dynamic_slippage(TradingDirection.LONG, current_price)
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
                leverage
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
            slippage = self._calculate_dynamic_slippage(TradingDirection.SHORT, current_price)
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
                leverage
            )
            self.time_in_position = 0
            self.pnl = 0.0
            self._update_liquidation_price(leverage)
            
        elif discrete_action == 3:  # Hold Position
            if self.position_state != 0:
                self.time_in_position += 1
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
                TradingDirection.LONG if self.position_state == 1 else TradingDirection.SHORT,
                current_price
            ),
            percentage
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
        self.pnl = TradingEnvUtils.calculate_pnl(
            current_price,
            self.position_open_price,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(
                TradingDirection.LONG if self.position_state == 1 else TradingDirection.SHORT,
                current_price
            )
        )

    def _close_position(self, current_price: float):
        self._update_pnl(current_price)
        closing_fee = TradingEnvUtils.calculate_close_fee(
            current_price,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(
                TradingDirection.LONG if self.position_state == 1 else TradingDirection.SHORT,
                current_price
            )
        )
        self.balance += self.pnl - closing_fee
        self.position_state = 0
        self.number_contracts_owned = 0.0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0.0
        self.liquidation_price = None
        self.current_leverage = 1.0

    def step(self, action):
        current_data = self.env_data_source.iloc[self.current_step]
        
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

        # Calculate reward using config factors
        reward = 0.0
        if self.was_liquidated:
            base_penalty = max(abs(self.last_liquidation_loss), self.env_config.min_liquidation_penalty)
            time_factor = max(self.time_in_position, 1)  # Ensure at least 1 time unit
            reward = -self.env_config.liquidation_penalty_factor * base_penalty * time_factor
            self.was_liquidated = False
            self.last_liquidation_loss = 0.0
        elif self.position_state != 0:
            if self.pnl > 0:
                reward = self.env_config.in_money_factor * self.pnl * np.sqrt(self.time_in_position)
            else:
                reward = -self.env_config.out_of_money_factor * abs(self.pnl) * (self.time_in_position ** 1.5)

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
