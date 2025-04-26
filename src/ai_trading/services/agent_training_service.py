from typing import Dict, Tuple

from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trading.agents.agent_factory import AgentFactory
from ai_trading.agents.base_agent import BaseAgent
from ai_trading.config.application_config import ApplicationConfig
from ai_trading.gyms.custom_env import TradingEnv
from ai_trading.model.split_dataset_container import SplitDataSetContainer


class AgentTrainingService:
    """
    Service to handle the creation of environments and training of agents.
    """

    def __init__(self, config: ApplicationConfig) -> None:
        """
        Initialize the training service with configuration and agent factory.

        Args:
            env_config: Environment configuration settings
        """
        self.config = config
        self.agent_factory = AgentFactory()

    def create_env_and_train_agents(
        self,
        final_datasets: list[SplitDataSetContainer],
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
            [
                *(
                    (
                        lambda dataset=dataset: TradingEnv(
                            dataset.training_data,
                            self.config.environment_config,
                            feature_start_index,
                        )
                    )
                    for dataset in final_datasets
                )
            ]
        )
        val_env = DummyVecEnv(
            [
                *(
                    (
                        lambda dataset=dataset: TradingEnv(
                            dataset.validation_data,
                            self.config.environment_config,
                            feature_start_index,
                        )
                    )
                    for dataset in final_datasets
                )
            ]
        )

        # Create agents using the factory
        agents = self.agent_factory.create_multiple_agents(
            self.config.rl_model_config.agents,
            train_env,
            self.config.rl_model_config.total_timesteps,
            self.config.rl_model_config.agent_threshold,
        )

        # Validate all agents
        for _agent_name, agent in agents.items():
            agent.validate(val_env)

        return train_env, val_env, agents
