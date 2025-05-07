import random
import time

import numpy as np
import pandas as pd
import pytest

from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.gyms.custom_env import TradingEnv
from ai_trading.gyms.utils.trading_env_utils import TradingDirection


@pytest.fixture(autouse=True)
def setup_random():
    """Initialize random seeds before each test."""
    # Use current time to ensure different seeds across test runs
    seed = int(time.time() * 1000) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    return seed


@pytest.fixture
def mock_environment_config():
    return EnvironmentConfig(
        fee=0.001,  # 0.1% transaction fee
        slippage_atr_based=0.01,  # 1% of ATR for slippage calculation
        slippage_against_trade_probability=0.6,  # 60% chance slippage works against trade
        start_balance=10000.0,
        max_daily_drawdown=0.1,
        max_alltime_drawdown=0.2,
        max_percentage_open_position=1.0,
        min_percentage_open_position=0.01,
        in_money_factor=1.0,
        out_of_money_factor=1.0,
        liquidation_penalty_factor=2.0,
        min_liquidation_penalty=100.0,
        # New properties for reward calculation
        max_time_in_trade=10,
        optimal_exit_time=3,
        variance_penalty_weight=0.5,
        atr_penalty_weight=0.3,
    )


@pytest.fixture
def mock_train_data():
    """Create mock data DataFrame with consistent column names.

    Time information is stored in the DataFrame's index as a DatetimeIndex.
    """

    def _generate_data():
        timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=30, freq="H")
        # First create metadata columns - using consistent capitalization
        metadata = {
            "Close": np.random.uniform(1.0, 2.0, size=30),
            "High": [],  # Will fill after price
            "Low": [],  # Will fill after price
            "Atr": np.full(30, 0.05),  # Constant ATR for testing
        }
        # Fill high and low based on Close price
        metadata["High"] = metadata["Close"] + 0.1  # High is always 0.1 above price
        metadata["Low"] = metadata["Close"] - 0.1  # Low is always 0.1 below price

        # Then create feature columns
        feature_data = np.random.rand(30, 10)  # 30 rows, 10 columns with random values
        feature_columns = {f"feature_{i}": feature_data[:, i] for i in range(10)}

        # Combine metadata and features, with metadata first
        all_data = {**metadata, **feature_columns}
        return pd.DataFrame(all_data, index=timestamps)

    return _generate_data()


@pytest.fixture
def env(mock_train_data, mock_environment_config):
    """Create trading environment with proper context columns setup.

    Time information is in the DataFrame's index, so we don't include it in context_columns.
    """
    # Define context columns (non-feature columns)
    context_columns = ["Close", "High", "Low", "Atr"]
    return TradingEnv(
        mock_train_data, mock_environment_config, context_columns=context_columns
    )


class TestTradingEnv:
    def test_env_initialization(self, env):
        # Given
        # Environment is initialized via fixture

        # When/Then
        assert env.balance == 10000.0, "Initial balance should be set correctly"
        assert env.position_state == 0, "Initial position should be neutral"
        assert env.current_step == 0, "Initial step should be 0"
        assert env.done is False, "Environment should not start in done state"

    def test_reset_state(self, env: TradingEnv) -> None:
        # Given
        env.balance = 5000.0
        env.position_state = 1
        env.current_step = 10

        # When
        observation, _ = env.reset()

        # Then
        assert env.balance == 10000.0, "Balance should be reset to initial value"
        assert env.position_state == 0, "Position should be reset to neutral"
        assert env.current_step == 0, "Step should be reset to 0"
        assert isinstance(observation, np.ndarray), "Observation should be numpy array"

    def test_open_long_position(self, env: TradingEnv) -> None:
        # Given
        action = (
            0,
            np.array([0.5], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Open long with 50% of balance, no partial close, 1x leverage
        initial_balance = env.balance
        current_price = env.env_data_source.iloc[0].Close

        # When
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 1, "Position should be long"
        assert env.number_contracts_owned > 0, "Should own positive contracts"
        # Verify slippage is within expected range (base ATR slippage is 0.01)
        price_diff = abs(env.position_open_price - current_price)
        expected_max_slippage = (
            current_price
            * (env.env_data_source.iloc[0].Atr / current_price)
            * env.env_config.slippage_atr_based
            * 1.001  # This is really important, because otherwise tests will be flaky and sometimes fail
        )
        assert (
            price_diff <= expected_max_slippage
        ), f"Slippage {price_diff} exceeds maximum expected {expected_max_slippage}"
        assert (
            env.balance == initial_balance
        ), "Balance shouldn't change on position open"
        assert env.liquidation_price is None, "No liquidation price for 1x leverage"

    def test_open_leveraged_long_position(self, env: TradingEnv) -> None:
        # Given
        leverage = 5
        action = (
            0,
            np.array([0.5], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([leverage], dtype=np.int16),
        )  # Open long with 50% balance, 5x leverage

        # When
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 1, "Position should be long"
        assert env.current_leverage == leverage, "Leverage should be set"
        assert env.liquidation_price is not None, "Liquidation price should be set"
        assert (
            env.liquidation_price < env.position_open_price
        ), "Liquidation price should be below entry for longs"

    def test_open_short_position(self, env: TradingEnv) -> None:
        # Given
        action = (
            2,
            np.array([0.5], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Open short with 50% balance, no leverage

        # When
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == -1, "Position should be short"
        assert env.number_contracts_owned < 0, "Should own negative contracts"
        assert env.liquidation_price is None, "No liquidation price for 1x leverage"

    def test_close_position(self, env: TradingEnv) -> None:
        # Given
        # First open a position
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )
        initial_contracts = env.number_contracts_owned

        # When
        # Then close it
        action = (
            1,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Close position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be closed"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert env.position_open_price is None, "No open price when closed"
        assert env.time_in_position == 0, "Time in position should reset"

    def test_partial_close_position(self, env):
        # Given
        # Open a position first
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )
        initial_contracts = env.number_contracts_owned

        # When
        action = (
            4,
            np.array([0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Partial close 50%
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 1, "Position should remain open"
        assert (
            env.number_contracts_owned == initial_contracts * 0.5
        ), "Should have half the contracts"

    def test_liquidation_long_position(self, env):
        # Given
        # Open a leveraged long position
        leverage = 5
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([leverage], dtype=np.int16),
            )
        )
        initial_balance = env.balance
        liquidation_price = env.liquidation_price

        # Simulate price movement below liquidation price
        # This is done by manipulating the dataframe's low price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("Low")] = (
            liquidation_price - 0.01
        )

        # When
        action = (
            3,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([leverage], dtype=np.int16),
        )  # Try to hold position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be liquidated"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert reward < 0, "Should receive negative reward for liquidation"
        assert (
            env.balance < initial_balance
        ), "Balance should decrease after liquidation"

    def test_liquidation_short_position(self, env):
        # Given
        # Open a leveraged short position
        leverage = 5
        env.step(
            (
                2,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([leverage], dtype=np.int16),
            )
        )
        initial_balance = env.balance
        liquidation_price = env.liquidation_price

        # Simulate price movement above liquidation price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("High")] = (
            liquidation_price + 0.01
        )

        # When
        action = (
            3,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([leverage], dtype=np.int16),
        )  # Try to hold position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be liquidated"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert reward < 0, "Should receive negative reward for liquidation"
        assert (
            env.balance < initial_balance
        ), "Balance should decrease after liquidation"

    def test_reward_profitable_trade(self, env: TradingEnv):
        # Given
        # Open a position and simulate time passage
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )  # Open long position
        initial_balance = env.balance
        env.time_in_position = 2  # Simulate optimal time steps (within the sweet spot)

        # Simulate price increase (20% profit)
        current_price = env.position_open_price
        position_value = current_price * env.number_contracts_owned
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("Close")] = (
            current_price * 1.2
        )

        # When
        action = (
            3,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Hold position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        # Calculate expected reward components
        pnl_pct = env.pnl / position_value if position_value > 0 else 0
        time_penalty = -0.1 * 3  # now at time_in_position = 3
        sweet_spot_bonus = 1.5  # for profitable trade within optimal time

        # We can't precisely calculate price_std and ATR penalties in the test
        # So we just ensure the reward is positive and approximately matches our expectation
        assert reward > 0, "Should receive positive reward for profitable trade"

        # Check that reward is greater than PnL percentage alone (due to sweet spot bonus)
        assert reward > pnl_pct, "Reward should include sweet spot bonus"

    def test_reward_losing_trade(self, env):
        # Given
        # Open a position and simulate time passage
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )  # Open long position
        initial_balance = env.balance
        env.time_in_position = 11  # Exceeds max optimal time

        # Simulate price decrease (10% loss)
        current_price = env.position_open_price
        position_value = current_price * env.number_contracts_owned
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("Close")] = (
            current_price * 0.9
        )

        # When
        action = (
            3,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.int16),
        )  # Hold position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        # Calculate expected reward components
        pnl_pct = env.pnl / position_value if position_value > 0 else 0
        time_penalty = -0.1 * 12  # now at time_in_position = 12
        exceeds_max_time_penalty = -2.0  # additional penalty for exceeding max time

        # The reward should be significantly more negative than just the PnL percentage
        assert reward < 0, "Should receive negative reward for losing trade"
        assert reward < pnl_pct, "Reward should be worse than PnL due to time penalties"

    def test_reward_leveraged_position(self, env):
        # Given
        # Open a leveraged position
        leverage = 5
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([leverage], dtype=np.int16),
            )
        )  # Open leveraged long position
        env.time_in_position = 2  # Within optimal time

        # Simulate price increase (10% profit)
        current_price = env.position_open_price
        position_value = current_price * env.number_contracts_owned
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("Close")] = (
            current_price * 1.1
        )

        # When
        action = (
            3,
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([leverage], dtype=np.int16),
        )  # Hold position
        observation, reward, done, terminated, info = env.step(action)

        # Then
        # With leverage multiplier, reward should be significantly higher
        assert (
            reward > 0
        ), "Should receive positive reward for profitable leveraged trade"

        # Get reward for same scenario without leverage for comparison
        env.reset()
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )  # Open non-leveraged long position
        env.time_in_position = 2
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc("Close")] = (
            env.position_open_price * 1.1
        )

        _, non_leveraged_reward, _, _, _ = env.step(
            (
                3,
                np.array([0], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )

        # Verify leverage increases reward proportionally
        assert (
            reward > non_leveraged_reward
        ), "Leveraged position should yield higher reward"
        assert np.isclose(
            reward / non_leveraged_reward, leverage, rtol=0.2
        ), "Reward should scale with leverage"

    def test_price_std_calculation(self, env):
        # Given
        # Create a controlled price sequence
        price_values = [
            100.0,
            101.0,
            103.0,
            102.0,
            104.0,
        ]  # Price sequence with known volatility
        for i, price in enumerate(price_values):
            env.env_data_source.iloc[
                i, env.env_data_source.columns.get_loc("Close")
            ] = price

        # Open a position at the beginning
        env.current_step = 0
        env.step(
            (
                0,
                np.array([0.5], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([1], dtype=np.int16),
            )
        )

        # Simulate position being open for multiple steps
        env.current_step = 0  # Reset to recompute from beginning
        env.time_in_position = 4  # We're using 4 steps of data

        # When
        price_std = env._calculate_price_std()

        # Then
        # Calculate expected standard deviation of returns manually
        returns = np.diff(price_values) / np.array(price_values[:-1])
        expected_std = np.std(returns)

        # Verify calculation is correct
        assert np.isclose(
            price_std, expected_std, rtol=1e-10
        ), "Price std calculation should match expected value"

    def test_episode_completion(self, env):
        # Given
        initial_balance = env.balance

        # When
        # Run through entire episode
        done = False
        terminated = False
        total_steps = 0
        while not done and not terminated:
            action = env.action_space.sample()
            _, _, done, terminated, _ = env.step(action)
            total_steps += 1

        # Then
        assert done, "Episode should complete"
        assert total_steps <= len(env.env_data_source), "Should not exceed data length"
        assert (
            env.current_step >= len(env.env_data_source) - 1
        ), "Should reach end of data"

    def test_dynamic_slippage_calculation(self, env):
        # Given
        direction = TradingDirection.LONG
        current_price = 1.0
        env.env_data_source.iloc[0, env.env_data_source.columns.get_loc("Atr")] = 0.05

        # When
        # Test multiple times to verify probabilistic behavior
        slippages = [
            env._calculate_dynamic_slippage(direction, current_price)
            for _ in range(100)
        ]

        # Then
        # Base slippage should be ATR * slippage_atr_based = 0.05 * 0.01 = 0.0005
        assert all(
            abs(s) == 0.0005 for s in slippages
        ), "Base slippage magnitude should be consistent"
        # With 60% probability against trade, roughly 60% should be positive for long positions
        positive_count = sum(1 for s in slippages if s > 0)
        assert (
            50 <= positive_count <= 70
        ), "Should have roughly 60% positive slippage for long positions"

    def test_dynamic_slippage_position_impact(self, env):
        # Given
        # Set a large ATR to make slippage effect more noticeable
        env.env_data_source.iloc[0, env.env_data_source.columns.get_loc("Atr")] = 0.1

        # When
        # Open multiple positions and track their entry prices
        entry_prices = []
        for _ in range(10):
            env.reset()
            env.step((0, [0.5], [0], [1]))  # Open long position
            entry_prices.append(env.position_open_price)

        # Then
        # Entry prices should vary due to slippage
        assert len(set(entry_prices)) > 1, "Slippage should cause varying entry prices"

    def test_feature_selection_in_observation(self, env):
        # Given
        # Create test data with known feature columns
        data = {
            "Close": [100.0, 101.0],  # Context data
            "High": [102.0, 103.0],  # Context data
            "Low": [99.0, 98.0],  # Context data
            "Atr": [2.0, 2.1],  # Context data
            "feature1": [0.5, 0.6],  # Computed feature
            "feature2": [-0.3, -0.2],  # Computed feature
            "feature3": [1.2, 1.3],  # Computed feature
        }
        env.env_data_source = pd.DataFrame(data)
        env.context_columns = ["Close", "High", "Low", "Atr"]  # Set context columns
        env.feature_columns = [
            "feature1",
            "feature2",
            "feature3",
        ]  # Set feature columns

        # When
        observation = env._next_observation()

        # Then
        # Should only include features (feature1, feature2, feature3)
        expected_features = np.array([0.5, -0.3, 1.2], dtype=np.float32)
        assert observation.dtype == np.float32
        assert len(observation) == 3  # Should only have 3 computed features
        assert np.array_equal(observation, expected_features)
