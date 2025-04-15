import pytest
from unittest.mock import MagicMock
from ai_trading.services.agent_training_service import AgentTrainingService
from stable_baselines3.common.vec_env import DummyVecEnv

@pytest.fixture
def mock_train_data():
    return MagicMock()

@pytest.fixture
def mock_val_data():
    return MagicMock()

@pytest.fixture
def mock_agent_config():
    return ["PPO", "A2C"]

def test_create_env_and_train_agents(mock_train_data, mock_val_data, mock_agent_config):
    # Arrange
    service = AgentTrainingService()
    total_timesteps = 10000
    threshold = 0.5

    # Act
    train_env, val_env, agents = service.create_env_and_train_agents(
        mock_train_data, mock_val_data, total_timesteps, threshold, mock_agent_config
    )

    # Assert
    assert isinstance(train_env, DummyVecEnv), "Train environment should be a DummyVecEnv."
    assert isinstance(val_env, DummyVecEnv), "Validation environment should be a DummyVecEnv."
    assert len(agents) == len(mock_agent_config), "Number of agents should match the configuration."
    for agent_name in mock_agent_config:
        assert agent_name in agents, f"Agent {agent_name} should be in the agents dictionary."