"""
Factory for creating agent instances based on configuration.
"""

from typing import Dict, List, Type

from stable_baselines3.common.vec_env import VecEnv

from drl_trading_training.core.agents.base_agent import BaseAgent
from drl_trading_training.core.agents.ppo_agent import PPOAgent


class AgentFactory:
    """
    Factory class responsible for creating agent instances based on configuration strings.

    This class follows the factory pattern to dynamically create agent instances based on
    string identifiers. It replaces the previous registry pattern with a simpler, more direct approach.
    """

    def __init__(self) -> None:
        """
        Initialize the agent factory with mappings of agent names to their classes.

        The mapping is kept simple and can be extended with more agent types as needed.
        """
        # Mapping of agent name strings to their respective classes
        self.agent_class_map: Dict[str, Type[BaseAgent]] = {
            "PPO": PPOAgent,
            # Add other agents here as they are implemented
            # "DQN": DQNAgent,
            # "A2C": A2CAgent,
            # etc.
        }

    def create_agent(
        self,
        agent_type: str,
        env: VecEnv,
        total_timesteps: int,
        threshold: float = 0.5,
    ) -> BaseAgent:
        """
        Create and return an agent instance of the specified type.

        Args:
            agent_type: String identifier for the agent type (e.g., "PPO")
            env: Vectorized environment for training
            total_timesteps: Number of timesteps to train for
            threshold: Decision threshold for action recommendations

        Returns:
            An instance of the requested agent

        Raises:
            ValueError: If the specified agent type is not supported
        """
        if agent_type not in self.agent_class_map:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        agent_class = self.agent_class_map[agent_type]
        # Using positional arguments instead of keyword arguments
        return agent_class(env, total_timesteps, threshold)

    def create_multiple_agents(
        self,
        agent_types: List[str],
        env: VecEnv,
        total_timesteps: int,
        threshold: float = 0.5,
    ) -> Dict[str, BaseAgent]:
        """
        Create multiple agents based on a list of agent type strings.

        Args:
            agent_types: List of agent type strings
            env: Vectorized environment for training
            total_timesteps: Number of timesteps to train for
            threshold: Decision threshold for action recommendations

        Returns:
            Dictionary mapping agent type strings to agent instances
        """
        agents: Dict[str, BaseAgent] = {}

        for agent_type in agent_types:
            if agent_type in self.agent_class_map:
                agents[agent_type] = self.create_agent(
                    agent_type, env, total_timesteps, threshold
                )

        return agents
