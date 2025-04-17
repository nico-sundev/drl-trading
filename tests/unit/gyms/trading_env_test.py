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
        data = np.random.rand(30, 10)  # 30 rows, 10 columns with random values between 0 and 1
        columns = [f"feature_{i}" for i in range(1, 11)]
        df = pd.DataFrame(data, index=timestamps, columns=columns)
        df['price'] = np.random.uniform(1.0, 2.0, size=30)  # Add price column
        df['high'] = df['price'] + 0.1  # High is always 0.1 above price
        df['low'] = df['price'] - 0.1   # Low is always 0.1 below price
        df['atr'] = 0.05  # Add ATR column with constant value for testing
        return df
    return _generate_data()  # Return a new DataFrame each time the fixture is used

@pytest.fixture
def env(mock_train_data, mock_environment_config):
    return TradingEnv(mock_train_data, mock_environment_config)

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

    def test_reward_profitable_trade(self, env):
        # Given
        # Open a position
        env.step((0, [0.5], [0], [1]))
        
        # Simulate price increase
        current_price = env.env_data_source.iloc[0].price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('price')] = current_price * 1.1

        # When
        action = (3, [0], [0], [1])  # Hold position
        observation, reward, done, info = env.step(action)

        # Then
        assert reward > 0, "Should receive positive reward for profitable trade"
        assert env.pnl > 0, "Should have positive PnL"

    def test_reward_losing_trade(self, env):
        # Given
        # Open a position
        env.step((0, [0.5], [0], [1]))
        
        # Simulate price decrease
        current_price = env.env_data_source.iloc[0].price
        env.env_data_source.iloc[1, env.env_data_source.columns.get_loc('price')] = current_price * 0.9

        # When
        action = (3, [0], [0], [1])  # Hold position
        observation, reward, done, info = env.step(action)

        # Then
        assert reward < 0, "Should receive negative reward for losing trade"
        assert env.pnl < 0, "Should have negative PnL"

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

    def test_observation_space(self, env):
        # Given
        observation = env.reset()

        # When/Then
        assert observation.shape[0] == 7, "Observation should include all state variables"
        assert isinstance(observation[0], pd.Series), "First element should be feature set"
        assert isinstance(observation[1], int), "Second element should be position state"
        assert isinstance(observation[2], int), "Third element should be time in position"
        assert isinstance(observation[3], float), "Fourth element should be PnL"
        assert isinstance(observation[4], float), "Fifth element should be number of contracts"

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