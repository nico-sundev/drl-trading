import numpy as np
import pandas as pd
import pytest

from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.custom_env import TradingEnv

@pytest.fixture
def mock_environment_config():
    return EnvironmentConfig(
        fee=0.001,  # 0.1% transaction fee
        slippage_atr_based=0.05,  # 5% slippage based on ATR
        start_balance=10000.0,  # Starting balance of $10,000
        max_percentage_open_position=0.05,  
        min_percentage_open_position=0.01,
        maxDailyDrawdown=0.1,  # 10% maximum daily drawdown
        maxAlltimeDrawdown=0.2,  # 20% maximum all-time drawdown
    )

@pytest.fixture
def mock_train_data():
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=30, freq="H")
    data = np.random.rand(30, 10)  # 30 rows, 10 columns with random values between 0 and 1
    columns = [f"feature_{i}" for i in range(1, 11)]
    df = pd.DataFrame(data, index=timestamps, columns=columns)
    df['price'] = np.random.uniform(1.0, 2.0, size=30)  # Add a 'price' column with random values between 1.0 and 2.0
    return df

def test_trading_env(mock_train_data, mock_environment_config):

    # Create the environment
    env = TradingEnv(mock_train_data, mock_environment_config)

    # Example of running a random agent
    observation = env.reset()
    all_done = False
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            all_done = True
            break  # Exit the loop when the episode is done
            
    assert all_done == True, "The environment should be able to complete the episode."