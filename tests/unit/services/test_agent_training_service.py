from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.services.agent_training_service import AgentTrainingService


@pytest.fixture
def mock_environment_config():
    return EnvironmentConfig(
        fee=0.001,  # 0.1% transaction fee
        slippage_atr_based=0.05,  # 5% slippage based on ATR
        start_balance=10000.0,  # Starting balance of $10,000
        max_daily_drawdown=0.02,  # 2% max daily drawdown
        max_alltime_drawdown=0.05,  # 5% max all-time drawdown
        max_percentage_open_position=100.0,
        min_percentage_open_position=1.0,
    )


@pytest.fixture
def mock_agent_registry():
    registry = MagicMock()
    # Mock agent class that returns a mock agent
    mock_agent_class = MagicMock()
    mock_agent_class.return_value = MagicMock(validate=MagicMock(return_value=True))

    # Set up the agent_class_map with our mock agent class
    registry.agent_class_map = {"PPO": mock_agent_class}
    return registry


@pytest.fixture
def mock_train_data():
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=30, freq="H")
    data = np.random.rand(
        30, 10
    )  # 30 rows, 10 columns with random values between 0 and 1
    columns = [f"feature_{i}" for i in range(1, 11)]
    return pd.DataFrame(data, index=timestamps, columns=columns)


@pytest.fixture
def mock_val_data():
    timestamps = pd.date_range(start="2025-01-02 00:00:00", periods=30, freq="H")
    data = np.random.rand(
        30, 10
    )  # 30 rows, 10 columns with random values between 0 and 1
    columns = [f"feature_{i}" for i in range(1, 11)]
    return pd.DataFrame(data, index=timestamps, columns=columns)


@pytest.fixture
def mock_agent_config():
    return ["PPO"]


def test_create_env_and_train_agents(
    mock_train_data,
    mock_val_data,
    mock_agent_config,
    mock_environment_config,
    mock_agent_registry,
):
    # Arrange
    service = AgentTrainingService(mock_environment_config, mock_agent_registry)
    total_timesteps = 10000
    threshold = 0.5

    # Act
    train_env, val_env, agents = service.create_env_and_train_agents(
        mock_train_data, mock_val_data, total_timesteps, threshold, mock_agent_config
    )

    # Assert
    assert isinstance(
        train_env, DummyVecEnv
    ), "Train environment should be a DummyVecEnv."
    assert isinstance(
        val_env, DummyVecEnv
    ), "Validation environment should be a DummyVecEnv."
    assert len(agents) == len(
        mock_agent_config
    ), "Number of agents should match the configuration."
    for agent_name in mock_agent_config:
        assert (
            agent_name in agents
        ), f"Agent {agent_name} should be in the agents dictionary."
