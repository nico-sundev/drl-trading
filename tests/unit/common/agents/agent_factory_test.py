from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from ai_trading.common.agents.agent_factory import AgentFactory
from ai_trading.common.agents.ppo_agent import PPOAgent


@pytest.fixture
def factory() -> AgentFactory:
    """Return an instance of the AgentFactory for testing."""
    return AgentFactory()


class MockVecEnv(VecEnv):
    """Mock version of a VecEnv for testing."""

    def __init__(self) -> None:
        super().__init__(1, np.zeros(10))

    def reset(self) -> np.ndarray:
        return np.zeros(10)

    def step_async(self, actions: np.ndarray) -> None:
        pass

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        return np.zeros(10), np.array([0.0]), np.array([False]), {}

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: List[int] = None) -> List[Any]:
        return [None]

    def set_attr(self, attr_name: str, value: Any, indices: List[int] = None) -> None:
        pass

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: List[int] = None,
        **method_kwargs: Any,
    ) -> List[Any]:
        return [None]

    def get_images(self, *args: Any, **kwargs: Any) -> List[np.ndarray]:
        return [np.zeros((10, 10, 3))]

    def env_is_wrapped(
        self, wrapper_class: type, indices: List[int] = None
    ) -> List[bool]:
        return [False]


@pytest.fixture
def mock_env() -> MockVecEnv:
    """Return a mock VecEnv for testing."""
    return MockVecEnv()


def test_factory_initialization(factory: AgentFactory) -> None:
    """Test that AgentFactory initializes properly with expected agent types."""
    # Given
    # Factory is created via fixture

    # When/Then
    assert "PPO" in factory.agent_class_map
    assert factory.agent_class_map["PPO"] == PPOAgent
    assert isinstance(factory.agent_class_map, dict)


def test_create_agent_with_valid_type(
    factory: AgentFactory, mock_env: MockVecEnv
) -> None:
    """Test creating a single agent with a valid type."""
    # Given
    total_timesteps = 100
    threshold = 0.5

    # Mock the agent initialization to avoid actual training
    with mock.patch.object(
        PPOAgent, "__init__", return_value=None
    ) as mock_init, mock.patch.object(
        PPOAgent, "predict", return_value=np.array([0.0])
    ) as mock_predict, mock.patch.object(
        PPOAgent, "validate"
    ) as mock_validate:

        # When
        agent = factory.create_agent("PPO", mock_env, total_timesteps, threshold)

        # Then
        mock_init.assert_called_once_with(mock_env, total_timesteps, threshold)
        assert isinstance(agent, PPOAgent)


def test_create_agent_with_invalid_type(
    factory: AgentFactory, mock_env: MockVecEnv
) -> None:
    """Test that creating an agent with an invalid type raises ValueError."""
    # Given
    # Mock env is created via fixture

    # When/Then
    with pytest.raises(ValueError, match="Unsupported agent type: InvalidAgent"):
        factory.create_agent("InvalidAgent", mock_env, 100)


def test_create_multiple_agents(factory: AgentFactory, mock_env: MockVecEnv) -> None:
    """Test creating multiple agents at once."""
    # Given
    total_timesteps = 100
    agent_types = ["PPO"]

    # Mock the agent initialization to avoid actual training
    with mock.patch.object(
        PPOAgent, "__init__", return_value=None
    ) as mock_init, mock.patch.object(
        PPOAgent, "predict", return_value=np.array([0.0])
    ) as mock_predict, mock.patch.object(
        PPOAgent, "validate"
    ) as mock_validate:

        # When
        agents = factory.create_multiple_agents(agent_types, mock_env, total_timesteps)

        # Then
        assert len(agents) == 1
        assert "PPO" in agents
        assert isinstance(agents["PPO"], PPOAgent)
        mock_init.assert_called_once_with(mock_env, total_timesteps, 0.5)


def test_create_multiple_agents_with_some_invalid(
    factory: AgentFactory, mock_env: MockVecEnv
) -> None:
    """Test creating multiple agents where some types are invalid."""
    # Given
    total_timesteps = 100
    agent_types = ["PPO", "InvalidAgent"]

    # Mock the agent initialization to avoid actual training
    with mock.patch.object(
        PPOAgent, "__init__", return_value=None
    ) as mock_init, mock.patch.object(
        PPOAgent, "predict", return_value=np.array([0.0])
    ) as mock_predict, mock.patch.object(
        PPOAgent, "validate"
    ) as mock_validate:

        # When
        agents = factory.create_multiple_agents(agent_types, mock_env, total_timesteps)

        # Then
        assert len(agents) == 1
        assert "PPO" in agents
        assert "InvalidAgent" not in agents
