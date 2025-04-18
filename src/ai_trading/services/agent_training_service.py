from typing import Dict, List, Tuple

from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.agent_collection import EnsembleAgent, PPOAgent
from ai_trading.agents.agent_registry import AgentRegistry
from ai_trading.config.environment_config import EnvironmentConfig
from ai_trading.gyms.custom_env import TradingEnv


class AgentTrainingService:
    """
    Service to handle the creation of environments and training of agents.
    """

    def __init__(
        self, env_config: EnvironmentConfig, agent_registry: AgentRegistry
    ) -> None:
        self.agent_registry = agent_registry
        self.env_config = env_config

    def create_env_and_train_agents(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        total_timesteps: int,
        threshold: float,
        agent_config: List[str],
    ) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, PPOAgent]]:
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
        # Create environments for training and validation
        train_env = DummyVecEnv([lambda: TradingEnv(train_data, self.env_config)])
        val_env = DummyVecEnv([lambda: TradingEnv(val_data, self.env_config)])

        agents: Dict[str, PPOAgent] = {}
        for agent_name in agent_config:
            if agent_name in self.agent_registry.agent_class_map:
                agent_class = self.agent_registry.agent_class_map[agent_name]
                agents[agent_name] = agent_class(train_env, total_timesteps, threshold)
                agents[agent_name].validate(val_env)

        # Create the ensemble agent if specified
        if "Ensemble" in agent_config:
            ensemble_agents = [
                agent for name, agent in agents.items() if name != "Ensemble"
            ]
            agents["Ensemble"] = EnsembleAgent(ensemble_agents, threshold)
            agents["Ensemble"].validate(val_env)

        return train_env, val_env, agents
