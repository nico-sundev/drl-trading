import numpy as np
import pandas as pd
import pytest
import time
import random

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
        min_liquidation_penalty=100.0
    )

@pytest.fixture
def mock_train_data():
    def _generate_data():
        timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=30, freq="H")
        # First create metadata columns
        metadata = {
            'price': np.random.uniform(1.0, 2.0, size=30),
            'high': [],  # Will fill after price
            'low': [],   # Will fill after price
            'atr': np.full(30, 0.05)  # Constant ATR for testing
        }
        # Fill high and low based on price
        metadata['high'] = metadata['price'] + 0.1  # High is always 0.1 above price
        metadata['low'] = metadata['price'] - 0.1   # Low is always 0.1 below price
        
        # Then create feature columns
        feature_data = np.random.rand(30, 10)  # 30 rows, 10 columns with random values
        feature_columns = {f'feature_{i}': feature_data[:, i] for i in range(10)}
        
        # Combine metadata and features, with metadata first
        all_data = {**metadata, **feature_columns}
        return pd.DataFrame(all_data, index=timestamps)
    
    return _generate_data()

@pytest.fixture
def env(mock_train_data, mock_environment_config):
    # Feature start index is 4 since we have price, high, low, atr as metadata columns
    return TradingEnv(mock_train_data, mock_environment_config, feature_start_index=4)

class TestTradingEnv:
    def test_env_initialization(self, env):
        # Given
        # Environment is initialized via fixture

        # When/Then
        assert env.balance == 10000.0, "Initial balance should be set correctly"
        assert env.position_state == 0, "Initial position should be neutral"
        assert env.current_step == 0, "Initial step should be 0"
        assert env.done is False, "Environment should not start in done state"

    def test_reset_state(self, env):
        # Given
        env.balance = 5000.0
        env.position_state = 1
        env.current_step = 10

        # When
        observation = env.reset()

        # Then
        assert env.balance == 10000.0, "Balance should be reset to initial value"
        assert env.position_state == 0, "Position should be reset to neutral"
        assert env.current_step == 0, "Step should be reset to 0"
        assert isinstance(observation, np.ndarray), "Observation should be numpy array"

    def test_open_long_position(self, env):
        # Given
        action = (0, [0.5], [0], [1])  # Open long with 50% of balance, no partial close, 1x leverage
        initial_balance = env.balance
        current_price = env.env_data_source.iloc[0].price
        
        # When
        observation, reward, done, info = env.step(action)
        
        # Then
        assert env.position_state == 1, "Position should be long"
        assert env.number_contracts_owned > 0, "Should own positive contracts"
        # Verify slippage is within expected range (base ATR slippage is 0.01)
        price_diff = abs(env.position_open_price - current_price)
        expected_max_slippage = current_price * (env.env_data_source.iloc[0].atr / current_price) * env.env_config.slippage_atr_based
        assert price_diff <= expected_max_slippage, f"Slippage {price_diff} exceeds maximum expected {expected_max_slippage}"
        assert env.balance == initial_balance, "Balance shouldn't change on position open"
        assert env.liquidation_price is None, "No liquidation price for 1x leverage"

    def test_open_leveraged_long_position(self, env):
        # Given
        leverage = 5
        action = (0, [0.5], [0], [leverage])  # Open long with 50% balance, 5x leverage

        # When
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == 1, "Position should be long"
        assert env.current_leverage == leverage, "Leverage should be set"
        assert env.liquidation_price is not None, "Liquidation price should be set"
        assert env.liquidation_price < env.position_open_price, "Liquidation price should be below entry for longs"

    def test_open_short_position(self, env):
        # Given
        action = (2, [0.5], [0], [1])  # Open short with 50% balance, no leverage

        # When
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == -1, "Position should be short"
        assert env.number_contracts_owned < 0, "Should own negative contracts"
        assert env.liquidation_price is None, "No liquidation price for 1x leverage"

    def test_close_position(self, env):
        # Given
        # First open a position
        env.step((0, [0.5], [0], [1]))
        initial_contracts = env.number_contracts_owned
        
        # When
        # Then close it
        action = (1, [0], [0], [1])  # Close position
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be closed"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert env.position_open_price is None, "No open price when closed"
        assert env.time_in_position == 0, "Time in position should reset"

    def test_partial_close_position(self, env):
        # Given
        # Open a position first
        env.step((0, [0.5], [0], [1]))
        initial_contracts = env.number_contracts_owned
        
        # When
        action = (4, [0], [0.5], [1])  # Partial close 50%
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == 1, "Position should remain open"
        assert env.number_contracts_owned == initial_contracts * 0.5, "Should have half the contracts"

    def test_liquidation_long_position(self, env):
        # Given
        # Open a leveraged long position
        leverage = 5
        env.step((0, [0.5], [0], [leverage]))
        initial_balance = env.balance
        liquidation_price = env.liquidation_price

        # Simulate price movement below liquidation price
        # This is done by manipulating the dataframe's low price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('low')] = liquidation_price - 0.01

        # When
        action = (3, [0], [0], [leverage])  # Try to hold position
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be liquidated"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert reward < 0, "Should receive negative reward for liquidation"
        assert env.balance < initial_balance, "Balance should decrease after liquidation"

    def test_liquidation_short_position(self, env):
        # Given
        # Open a leveraged short position
        leverage = 5
        env.step((2, [0.5], [0], [leverage]))
        initial_balance = env.balance
        liquidation_price = env.liquidation_price

        # Simulate price movement above liquidation price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('high')] = liquidation_price + 0.01

        # When
        action = (3, [0], [0], [leverage])  # Try to hold position
        observation, reward, done, info = env.step(action)

        # Then
        assert env.position_state == 0, "Position should be liquidated"
        assert env.number_contracts_owned == 0, "Should own no contracts"
        assert reward < 0, "Should receive negative reward for liquidation"
        assert env.balance < initial_balance, "Balance should decrease after liquidation"

    def test_reward_profitable_trade(self, env: TradingEnv):
        # Given
        # Open a position and simulate time passage
        env.step((0, [0.5], [0], [1]))  # Open long position
        initial_balance = env.balance
        env.time_in_position = 4  # Simulate 4 time steps
        
        # Simulate price increase (20% profit)
        current_price = env.position_open_price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('price')] = current_price * 1.2

        # When
        action = (3, [0], [0], [1])  # Hold position
        observation, reward, done, info = env.step(action)

        # Then
        # With in_money_factor=1.0, sqrt(time)=2, and ~20% profit
        # Reward should be approximately 0.2 * initial_balance * 0.5 * 2
        expected_pnl = env.pnl
        expected_reward = env.env_config.in_money_factor * expected_pnl * np.sqrt(5)
        assert np.isclose(reward, expected_reward, rtol=0.1)
        assert reward > 0, "Should receive positive reward for profitable trade"

    def test_reward_losing_trade(self, env):
        # Given
        # Open a position and simulate time passage
        env.step((0, [0.5], [0], [1]))  # Open long position
        initial_balance = env.balance
        env.time_in_position = 2  # Simulate 2 time steps
        
        # Simulate price decrease (10% loss)
        current_price = env.position_open_price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('price')] = current_price * 0.9

        # When
        action = (3, [0], [0], [1])  # Hold position
        observation, reward, done, info = env.step(action)

        # Then
        # With out_of_money_factor=1.0, time^1.5~2.83, and ~10% loss
        # Penalty should be approximately 0.1 * initial_balance * 0.5 * 2.83
        expected_pnl = env.pnl
        expected_reward = -env.env_config.out_of_money_factor * abs(expected_pnl) * (3 ** 1.5)
        assert np.isclose(reward, expected_reward, rtol=0.1)
        assert reward < 0, "Should receive negative reward for losing trade"

    def test_reward_liquidation(self, env):
        # Given
        # Open a leveraged position
        leverage = 5
        env.step((0, [0.5], [0], [leverage]))
        initial_balance = env.balance
        liquidation_price = env.liquidation_price
        env.time_in_position = 3  # Simulate some time in position

        # Simulate price movement below liquidation
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('low')] = liquidation_price - 0.01

        # When
        action = (3, [0], [0], [leverage])  # Try to hold position
        observation, reward, done, info = env.step(action)

        # Then
        # With liquidation_penalty_factor=2.0, min_penalty=100, and time=3
        # Penalty should be max(actual_loss, 100) * 2 * 3
        min_penalty = env.env_config.min_liquidation_penalty * env.env_config.liquidation_penalty_factor * 4
        assert reward <= -min_penalty, "Should receive at least minimum liquidation penalty"
        assert reward < 0, "Should receive negative reward for liquidation"

    def test_episode_completion(self, env):
        # Given
        initial_balance = env.balance

        # When
        # Run through entire episode
        done = False
        total_steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            total_steps += 1

        # Then
        assert done, "Episode should complete"
        assert total_steps <= len(env.env_data_source), "Should not exceed data length"
        assert env.current_step >= len(env.env_data_source) - 1, "Should reach end of data"

    def test_dynamic_slippage_calculation(self, env):
        # Given
        direction = TradingDirection.LONG
        current_price = 1.0
        env.env_data_source.iloc[0, env.env_data_source.columns.get_loc('atr')] = 0.05

        # When
        # Test multiple times to verify probabilistic behavior
        slippages = [env._calculate_dynamic_slippage(direction, current_price) for _ in range(100)]

        # Then
        # Base slippage should be ATR * slippage_atr_based = 0.05 * 0.01 = 0.0005
        assert all(abs(s) == 0.0005 for s in slippages), "Base slippage magnitude should be consistent"
        # With 60% probability against trade, roughly 60% should be positive for long positions
        positive_count = sum(1 for s in slippages if s > 0)
        assert 50 <= positive_count <= 70, "Should have roughly 60% positive slippage for long positions"

    def test_dynamic_slippage_position_impact(self, env):
        # Given
        # Set a large ATR to make slippage effect more noticeable
        env.env_data_source.iloc[0, env.env_data_source.columns.get_loc('atr')] = 0.1
        
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
            'price': [100.0, 101.0],  # Strategy data
            'high': [102.0, 103.0],   # Strategy data
            'low': [99.0, 98.0],      # Strategy data
            'atr': [2.0, 2.1],        # Strategy data
            'feature1': [0.5, 0.6],   # Computed feature
            'feature2': [-0.3, -0.2],  # Computed feature
            'feature3': [1.2, 1.3]     # Computed feature
        }
        env.env_data_source = pd.DataFrame(data)
        env.feature_start_index = 4  # Start from 'feature1'
        
        # When
        observation = env._next_observation()
        
        # Then
        # Should only include features starting from index 4 (feature1, feature2, feature3)
        expected_features = np.array([0.5, -0.3, 1.2], dtype=np.float32)
        assert observation.dtype == np.float32
        assert len(observation) == 3  # Should only have 3 computed features
        assert np.array_equal(observation, expected_features)