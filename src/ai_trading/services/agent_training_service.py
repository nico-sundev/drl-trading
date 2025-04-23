from typing import Dict, List, Tuple

from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.agent_factory import AgentFactory
from ai_trading.agents.base_agent import BaseAgent
from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.gyms.custom_env import TradingEnv


class AgentTrainingService:
    """
    Service to handle the creation of environments and training of agents.
    """

    def __init__(self, env_config: EnvironmentConfig) -> None:
        """
        Initialize the training service with configuration and agent factory.

        Args:
            env_config: Environment configuration settings
        """
        self.env_config = env_config
        self.agent_factory = AgentFactory()

    def create_env_and_train_agents(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        total_timesteps: int,
        threshold: float,
        agent_config: List[str],
    ) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
        """
        Create environments and train agents dynamically based on the configuration.

        Args:
            train_data (pd.DataFrame): Training data.
            val_data (pd.DataFrame): Validation data.
            total_timesteps (int): Total timesteps for training.
            threshold (float): Threshold for agent validation.
            agent_config (List[str]): List of agent names to train.

        Returns:
            tuple: Training environment, validation environment, and trained agents.
        """
        # Assume feature columns start from index 5 (you might need to adjust this based on your data structure)
        feature_start_index = 5

        # Create environments for training and validation
        train_env = DummyVecEnv(
            [lambda: TradingEnv(train_data, self.env_config, feature_start_index)]
        )
        val_env = DummyVecEnv(
            [lambda: TradingEnv(val_data, self.env_config, feature_start_index)]
        )

        # Create agents using the factory
        agents = self.agent_factory.create_multiple_agents(
            agent_config, train_env, total_timesteps, threshold
        )

        # Validate all agents
        for _agent_name, agent in agents.items():
            agent.validate(val_env)

        return train_env, val_env, agents
