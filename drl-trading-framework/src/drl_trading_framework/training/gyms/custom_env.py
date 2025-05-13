import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas import DataFrame

from drl_trading_framework.common.config.environment_config import EnvironmentConfig
from drl_trading_framework.common.trading_constants import ALL_CONTEXT_COLUMNS
from drl_trading_framework.training.gyms.utils.trading_env_utils import (
    TradingDirection,
    TradingEnvUtils,
)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Single trading environment based on gym.Env.

    This environment can be used directly with gymnasium or wrapped in a VecEnv.
    """

    def __init__(
        self,
        env_data_source: DataFrame,
        env_config: EnvironmentConfig,
        context_columns: Optional[List[str]] = None,
    ):
        """Initialize the trading environment.

        Args:
            env_data_source: DataFrame containing price data and computed features
            env_config: Configuration for the trading environment
            context_columns: List of column names that are context columns (not features for the observation space)
                             If not provided, will use ALL_CONTEXT_COLUMNS
        """
        super(TradingEnv, self).__init__()
        self.env_config = env_config
        self.env_data_source = env_data_source

        # Determine context columns and feature columns
        if context_columns is not None:
            self.context_columns = context_columns
        else:
            # Default to all known context columns that exist in the DataFrame
            self.context_columns = [
                col for col in ALL_CONTEXT_COLUMNS if col in env_data_source.columns
            ]

        # Compute feature columns (all columns that are not context columns)
        self.feature_columns = [
            col for col in env_data_source.columns if col not in self.context_columns
        ]

        logger.debug(f"Context columns: {self.context_columns}")
        logger.debug(f"Feature columns: {self.feature_columns}")

        # Initialize from environment config
        self.initial_balance = env_config.start_balance
        self.balance = env_config.start_balance

        # Position tracking
        self.current_step = 0
        self.done = False
        self.terminated = False
        self.position_state = 0  # 1 for long, -1 for short, 0 for no position
        self.time_in_position = 0
        self.position_open_price: Optional[float] = None
        self.number_contracts_owned = 0.0
        self.pnl = 0.0
        self.current_leverage = 1.0
        self.liquidation_price: Optional[float] = None
        self.was_liquidated = False
        self.last_liquidation_loss = 0.0
        self.atr_at_entry = None

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

        # Validate if the mandatory DataFrame columns are present
        self._validate_columns()

        # Observation space calculation based on computed features only
        num_features = len(self.feature_columns)
        if num_features == 0:
            raise ValueError("No feature columns found for observation space")

        low = np.full(num_features, -np.inf)
        high = np.full(num_features, np.inf)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _validate_columns(self) -> None:
        """
        Validate that the DataFrame contains the mandatory columns.

        Time information can be either a column named "Time" or the DataFrame's index.
        Only validates primary context columns that should exist in raw data,
        not derived columns that are computed.
        """
        from drl_trading_framework.common.trading_constants import (
            PRIMARY_CONTEXT_COLUMNS,
            TIME_COLUMN,
        )

        # First check if time information is available (either as column or index)
        has_time_information = (
            TIME_COLUMN in self.env_data_source.columns
            or isinstance(self.env_data_source.index, pd.DatetimeIndex)
            or self.env_data_source.index.name == TIME_COLUMN
        )

        if not has_time_information:
            raise ValueError(
                f"DataFrame is missing time information. It should either have a '{TIME_COLUMN}' column "
                f"or use a DatetimeIndex."
            )

        # Then check other required columns
        missing_columns = [
            col
            for col in PRIMARY_CONTEXT_COLUMNS
            if col not in self.env_data_source.columns
        ]

        if missing_columns:
            raise ValueError(
                f"DataFrame is missing mandatory columns: {', '.join(missing_columns)}"
            )

        # Verify ATR (derived column) is present as it's required for trading logic
        if "Atr" not in self.env_data_source.columns:
            raise ValueError(
                "DataFrame is missing required derived column: Atr. "
                "Ensure the ContextFeatureService has computed this column before creating the environment."
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            np.ndarray: Initial observation after reset
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_state = 0
        self.number_contracts_owned = 0.0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0.0
        self.done = False
        self.terminated = False
        self.current_leverage = 1.0
        self.liquidation_price = None
        self.was_liquidated = False
        self.last_liquidation_loss = 0.0
        return self._next_observation(), options or {}

    def _next_observation(self) -> np.ndarray[np.float32, Any]:
        """Get the next observation from the environment.

        Returns only the computed and normalized features, excluding strategy-specific
        data like prices, ATR, etc.

        Returns:
            np.ndarray: Array of computed features in float32 format
        """
        feature_set = self.env_data_source.iloc[self.current_step]

        # Select only computed features using feature column names
        computed_features = feature_set[self.feature_columns].values

        return np.asarray(computed_features, dtype=np.float32)

    def _update_liquidation_price(self, leverage: float) -> None:
        """Update liquidation price for leveraged positions."""
        if leverage > 1.0 and self.position_state != 0:
            if self.position_open_price is not None:
                self.liquidation_price = TradingEnvUtils.calculate_liquidation_price(
                    self.position_open_price, self.number_contracts_owned, leverage
                )
            else:
                self.liquidation_price = None
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

    def _handle_liquidation(self, current_price: float) -> None:
        """Handle liquidation of a position."""
        logger.info(f"Position liquidated at price {current_price}")
        # Then close the position
        self._close_position(current_price)

    def _calculate_dynamic_slippage(
        self, direction: TradingDirection, current_price: float
    ) -> float:
        """Calculate dynamic slippage based on ATR and trade direction using utility method."""
        current_data = self.env_data_source.iloc[self.current_step]
        return TradingEnvUtils.calculate_dynamic_slippage(
            direction,
            current_price,
            current_data.Atr,
            self.env_config.slippage_atr_based,
            self.env_config.slippage_against_trade_probability,
        )

    def _calculate_price_std(self) -> float:
        """Calculate the standard deviation of price movement during the trade using utility method."""
        if self.time_in_position <= 1 or self.position_open_price is None:
            return 0.0

        # Get price history for the duration of the trade
        start_idx = max(0, self.current_step - self.time_in_position)
        end_idx = self.current_step + 1  # inclusive
        price_history = self.env_data_source.iloc[
            start_idx:end_idx
        ].Close.values.tolist()

        return TradingEnvUtils.calculate_price_std(price_history)

    def _calculate_reward(self) -> float:
        """Calculate reward for the current position state using an enhanced strategy.

        The reward function considers:
        1. PnL percentage (not absolute value)
        2. Time in trade with penalties for exceeding optimal durations
        3. Volatility penalties
        4. ATR-based risk penalties

        Returns:
            float: The calculated reward value
        """
        if (
            not TradingEnvUtils.has_active_position(self.position_state)
            or self.position_open_price is None
        ):
            return 0.0

        # Calculate PnL percentage instead of absolute value
        position_value = abs(self.position_open_price * self.number_contracts_owned)
        pnl_pct = self.pnl / position_value if position_value > 0 else 0.0

        # Calculate price standard deviation during trade
        price_std = self._calculate_price_std()

        # Configuration parameters (could be moved to environment config)
        max_time_in_trade = self.env_config.max_time_in_trade
        optimal_exit_time = self.env_config.optimal_exit_time
        variance_penalty_weight = self.env_config.variance_penalty_weight
        atr_penalty_weight = self.env_config.atr_penalty_weight

        # 1. Base reward on realized PnL percentage
        reward = pnl_pct

        # 2. Apply time penalty: soft linear penalty plus hard cutoff
        time_penalty = -0.1 * self.time_in_position
        if self.time_in_position > max_time_in_trade:
            time_penalty -= 2.0  # Sharp penalty if held too long

        reward += time_penalty

        # 3. Add bonus for exiting at the "sweet spot" time
        if pnl_pct > 0 and self.time_in_position <= optimal_exit_time:
            reward += 1.5  # Bonus for quick profitable trades

        # 4. Apply volatility (variance) penalty
        reward -= variance_penalty_weight * price_std

        # 5. Apply ATR-scaled risk penalty
        if self.atr_at_entry is not None:
            reward -= atr_penalty_weight * self.atr_at_entry

        # Scale by leverage to account for risk taken
        reward *= self.current_leverage

        return float(reward)

    def _take_action(
        self, action: Tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        discrete_action = action[0]
        open_position_percentage = action[1][0]
        partial_close_percentage = action[2][0]
        leverage = action[3][0]

        feature_set = self.env_data_source.iloc[self.current_step]
        current_price = feature_set["Close"]

        if discrete_action == 0:  # Open Long
            if self.position_state != 0:
                self._close_position(current_price)

            self.position_state = 1
            slippage = self._calculate_dynamic_slippage(
                TradingDirection.LONG, current_price
            )
            # Apply slippage to entry price
            self.position_open_price = current_price * (1 + slippage)

            # Save ATR at entry for risk calculation
            self.atr_at_entry = feature_set["Atr"]

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

            # Save ATR at entry for risk calculation
            self.atr_at_entry = feature_set["Atr"]

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

    def _partial_close_position(self, current_price: float, percentage: float) -> None:
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

    def _calculate_risk_adjusted_pnl(self, raw_pnl: float) -> float:
        """Calculates a Sharpe ratio-like risk-adjusted PnL using utility method."""
        if self.time_in_position <= 1:
            return raw_pnl

        # Get volatility measure during the trade
        price_std = self._calculate_price_std()

        # Get current ATR as a scaling factor
        current_data = self.env_data_source.iloc[self.current_step]

        return TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, self.current_leverage, current_data.Atr
        )

    def _update_pnl(self, current_price: float) -> None:
        """Update the position PnL with risk adjustment.

        Calculates raw PnL and then applies risk adjustment to make it comparable
        across different securities with varying volatility profiles.

        Args:
            current_price: Current asset price
        """
        direction = TradingEnvUtils.get_position_direction(self.position_state)
        raw_pnl = TradingEnvUtils.calculate_pnl(
            current_price,
            self.position_open_price if self.position_open_price is not None else 0.0,
            self.number_contracts_owned,
            self.env_config.fee,
            self._calculate_dynamic_slippage(direction, current_price),
        )

        # For now, we store the raw PnL value for backward compatibility
        # but we could switch to using risk-adjusted PnL if desired
        # self.pnl = raw_pnl

        # Calculate and store risk-adjusted PnL
        self.pnl = self._calculate_risk_adjusted_pnl(raw_pnl)

    def _close_position(self, current_price: float) -> None:
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
        self.atr_at_entry = None

    def step(
        self, action: Tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Tuple containing (discrete_action, open_position_percentage, partial_close_percentage, leverage)

        Returns:
            Tuple containing (observation, reward, done, info)
        """
        current_data = self.env_data_source.iloc[self.current_step]

        if TradingEnvUtils.has_active_position(self.position_state):
            self.time_in_position += 1

        # Check for liquidation before taking action
        if self._check_liquidation(current_data.High, current_data.Low):
            self._handle_liquidation(
                current_data.Low if self.position_state == 1 else current_data.High
            )
        else:
            self._take_action(action)

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.env_data_source) - 1:
            self.done = True

        # TODO Check if Drawdown is too high and set terminated to True
        if self.current_step >= len(self.env_data_source) - 1:
            self.terminated = True

        # Calculate reward
        reward = self._calculate_reward()

        # Reset after reward calculation
        if not TradingEnvUtils.has_active_position(self.position_state):
            self.pnl = 0.0
            self.time_in_position = 0
            self.liquidation_price = None

        return self._next_observation(), reward, self.done, self.terminated, {}

    def render(self, mode: str = "human") -> None:
        """Render the environment.

        Args:
            mode: The rendering mode
        """
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
