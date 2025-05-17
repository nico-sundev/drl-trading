"""
Unit tests for the AgentFactory class.

Tests the functionality of creating various agent instances with the factory pattern.
"""

import unittest.mock as mock

import pytest
from stable_baselines3.common.vec_env import VecEnv

from drl_trading_framework.common.agents.agent_factory import AgentFactory
from drl_trading_framework.common.agents.ppo_agent import PPOAgent


class TestAgentFactory:
    """Tests for the AgentFactory class."""

    def test_factory_initialization(self):
        """Test that the factory initializes with the expected agent mappings."""
        # Given
        # When
        factory = AgentFactory()

        # Then
        assert "PPO" in factory.agent_class_map
        assert factory.agent_class_map["PPO"] == PPOAgent

    @mock.patch.object(PPOAgent, "__init__", return_value=None)
    @mock.patch.object(PPOAgent, "predict")
    def test_create_agent_with_valid_type(self, mock_predict, mock_init):
        """Test creating an agent with a valid agent type."""
        # Given
        factory = AgentFactory()
        mock_env = mock.MagicMock(spec=VecEnv)
        total_timesteps = 100
        threshold = 0.7

        # When
        agent = factory.create_agent("PPO", mock_env, total_timesteps, threshold)

        # Then
        assert isinstance(agent, PPOAgent)
        mock_init.assert_called_once_with(mock_env, total_timesteps, threshold)

    def test_create_agent_with_invalid_type(self):
        """Test creating an agent with an invalid agent type raises ValueError."""
        # Given
        factory = AgentFactory()
        mock_env = mock.MagicMock(spec=VecEnv)
        total_timesteps = 100

        # When/Then
        with pytest.raises(ValueError, match="Unsupported agent type: InvalidAgent"):
            factory.create_agent("InvalidAgent", mock_env, total_timesteps)

    @mock.patch.object(PPOAgent, "__init__", return_value=None)
    @mock.patch.object(PPOAgent, "predict")
    def test_create_multiple_agents(self, mock_predict, mock_init):
        """Test creating multiple agents from a list of types."""
        # Given
        factory = AgentFactory()
        mock_env = mock.MagicMock(spec=VecEnv)
        total_timesteps = 100
        threshold = 0.7
        agent_types = ["PPO"]

        # When
        agents = factory.create_multiple_agents(
            agent_types, mock_env, total_timesteps, threshold
        )

        # Then
        assert len(agents) == 1
        assert "PPO" in agents
        assert isinstance(agents["PPO"], PPOAgent)
        mock_init.assert_called_once_with(mock_env, total_timesteps, threshold)

    @mock.patch.object(PPOAgent, "__init__", return_value=None)
    @mock.patch.object(PPOAgent, "predict")
    def test_create_multiple_agents_with_some_invalid(self, mock_predict, mock_init):
        """Test creating multiple agents where some agent types are invalid."""
        # Given
        factory = AgentFactory()
        mock_env = mock.MagicMock(spec=VecEnv)
        total_timesteps = 100
        threshold = 0.7
        agent_types = ["PPO", "InvalidAgent"]

        # When
        agents = factory.create_multiple_agents(
            agent_types, mock_env, total_timesteps, threshold
        )

        # Then
        assert len(agents) == 1
        assert "PPO" in agents
        assert "InvalidAgent" not in agents
        assert isinstance(agents["PPO"], PPOAgent)
        mock_init.assert_called_once_with(mock_env, total_timesteps, threshold)
