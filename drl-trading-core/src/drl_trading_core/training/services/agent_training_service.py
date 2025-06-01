from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

from drl_trading_common.base import BaseTradingEnv
from drl_trading_common.config.application_config import ApplicationConfig
from injector import inject
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_trading_core.common.agents.agent_factory import AgentFactory
from drl_trading_core.common.agents.base_agent import BaseAgent
from drl_trading_core.common.model.split_dataset_container import (
    SplitDataSetContainer,
)


class AgentTrainingServiceInterface(ABC):
    """
    Interface for services that handle the creation of environments and training of agents.

    This interface defines the contract for creating training and validation environments
    and coordinating the training of multiple agents.
    """

    @abstractmethod
    def create_env_and_train_agents(
        self, final_datasets: list[SplitDataSetContainer]
    ) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
        """
        Create environments and train agents dynamically based on the configuration.

        Args:
            final_datasets: List of split dataset containers for training
            env_class: The trading environment class to instantiate

        Returns:
            tuple: Training environment, validation environment, and trained agents
        """
        pass


class AgentTrainingService(AgentTrainingServiceInterface):
    """
    Service to handle the creation of environments and training of agents.
    """
    @inject
    def __init__(self, config: ApplicationConfig, env_class: Type[BaseTradingEnv]) -> None:
        """
        Initialize the training service with configuration and agent factory.

        Args:
            config: Application configuration settings
            env_class: The trading environment class to use
        """
        self.config = config
        self.agent_factory = AgentFactory()
        self.env_class = env_class

    def create_env_and_train_agents(
        self, final_datasets: list[SplitDataSetContainer]
    ) -> Tuple[DummyVecEnv, DummyVecEnv, Dict[str, BaseAgent]]:
        """
        Create environments and train agents dynamically based on the configuration.

        Args:
            final_datasets: List of split dataset containers for training

        Returns:
            tuple: Training environment, validation environment, and trained agents.
        """        # Create environments for training and validation
        env_class = self.env_class
        train_env = DummyVecEnv(
            [
                *(
                    (
                        lambda dataset=dataset: env_class(
                            dataset.training_data,
                            self.config.environment_config,
                            self.config.context_feature_config.get_all_context_columns(),
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
                        lambda dataset=dataset: env_class(
                            dataset.training_data,
                            self.config.environment_config,
                            self.config.context_feature_config.get_all_context_columns(),
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
