from unittest import mock

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from ai_trading.agents.abstract_base_agent import AbstractBaseAgent
from ai_trading.agents.agent_factory import AgentFactory
from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.services.agent_training_service import AgentTrainingService


# Mock implementation of AbstractBaseAgent for testing
class MockAgent(AbstractBaseAgent):
    """Mock agent implementation for testing"""

    def __init__(
        self, env: VecEnv, total_timesteps: int, threshold: float = 0.5
    ) -> None:
        """Initialize mock agent"""
        self.env = env
        self.total_timesteps = total_timesteps
        self.threshold = threshold
        self.validate_called = False

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Mock predict implementation"""
        return np.array([0.1])

    def validate(self, env: VecEnv) -> None:
        """Mock validate implementation that tracks calls"""
        self.validate_called = True


@pytest.fixture
def env_config() -> EnvironmentConfig:
    """Create a sample environment config for testing"""
    config = EnvironmentConfig()
    config.fee = 0.001
    config.slippage_atr_based = 0.1
    config.slippage_against_trade_probability = 0.5
    config.start_balance = 10000.0
    config.max_daily_drawdown = 0.05
    config.max_alltime_drawdown = 0.2
    config.max_percentage_open_position = 1.0
    config.min_percentage_open_position = 0.1
    config.in_money_factor = 1.0
    config.out_of_money_factor = 1.0
    return config


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing"""
    # Create a simple DataFrame with required columns for TradingEnv
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=100),
            "open": np.random.random(100) * 100,
            "high": np.random.random(100) * 100,
            "low": np.random.random(100) * 100,
            "close": np.random.random(100) * 100,
            "volume": np.random.random(100) * 1000,
            "feature1": np.random.random(100),
            "feature2": np.random.random(100),
        }
    )
    return df


@pytest.fixture
def training_service(env_config: EnvironmentConfig) -> AgentTrainingService:
    """Return an initialized training service instance"""
    return AgentTrainingService(env_config)


def test_agent_training_service_init(
    training_service: AgentTrainingService, env_config: EnvironmentConfig
) -> None:
    """Test that the AgentTrainingService initializes properly"""
    # Given
    # Training service is created via fixture

    # Then
    assert training_service.env_config == env_config
    assert isinstance(training_service.agent_factory, AgentFactory)


def test_create_env_and_train_agents(
    training_service: AgentTrainingService, sample_data: pd.DataFrame
) -> None:
    """Test the create_env_and_train_agents method"""
    # Given
    train_data = sample_data.copy()
    val_data = sample_data.copy()
    total_timesteps = 100
    threshold = 0.5
    agent_config = ["PPO"]

    # Mock the agent factory and environment creation
    with mock.patch.object(
        AgentFactory, "create_multiple_agents"
    ) as mock_create_agents, mock.patch.object(
        DummyVecEnv, "__init__", return_value=None
    ) as mock_env_init, mock.patch.object(
        DummyVecEnv, "reset"
    ), mock.patch.object(
        DummyVecEnv, "step"
    ):

        # Set up the mock to return a dictionary with one agent
        mock_agent = MockAgent(None, total_timesteps, threshold)
        mock_create_agents.return_value = {"PPO": mock_agent}

        # When
        train_env, val_env, agents = training_service.create_env_and_train_agents(
            train_data, val_data, total_timesteps, threshold, agent_config
        )

        # Then
        assert isinstance(train_env, DummyVecEnv)
        assert isinstance(val_env, DummyVecEnv)
        assert "PPO" in agents
        assert agents["PPO"] == mock_agent
        assert mock_agent.validate_called
        mock_create_agents.assert_called_once_with(
            agent_config, train_env, total_timesteps, threshold
        )
