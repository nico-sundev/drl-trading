from typing import Any, Generic, Type, TypeVar

import numpy as np
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from ai_trading.agents.abstract_base_agent import AbstractBaseAgent
from ai_trading.policies.pol_grad_loss_cb import PolicyGradientLossCallback

# Define a TypeVar bound to BaseAlgorithm for better typing
ModelType = TypeVar("ModelType", bound=BaseAlgorithm)


class AgentPolicy(AbstractBaseAgent, Generic[ModelType]):
    """
    Base agent policy class that implements common functionality for all RL agent types.
    Concrete agent implementations will extend this class and specify their model type.
    """

    model: ModelType  # Type hint for the model attribute

    def __init__(self, env: Env, total_timesteps: int, threshold: float = 0.5) -> None:
        """
        Initialize an agent policy with specified model class.

        Args:
            env: Training environment
            model_class: Class of the model to use (PPO, A2C, DDPG, SAC, TD3, etc.)
            total_timesteps: Total timesteps for training
            threshold: Threshold for action recommendations
        """
        self.callback = PolicyGradientLossCallback()
        # The specific model instantiation will be handled by subclasses
        self.model = self._create_model(env, type(ModelType))
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        self.threshold = threshold

    def _create_model(self, env: Env, model_class: Type[ModelType]) -> ModelType:
        """
        Create a model instance with the given environment.
        This method should be overridden by subclasses if they need custom model creation.

        Args:
            env: Training environment
            model_class: Class of the model to use

        Returns:
            Instantiated model
        """
        # Call the constructor directly, letting the class handle its own default parameters
        return model_class("MlpPolicy", env, verbose=1)

    def predict(self, obs: Any) -> np.ndarray:
        """
        Generate a prediction using the trained model.

        Args:
            obs: Observation from the environment

        Returns:
            Action as numpy array
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def validate(self, env: Env) -> None:
        """
        Validate the agent's performance on the given environment.

        Args:
            env: Gymnasium environment to validate against
        """
        obs, _ = env.reset()
        total_rewards = 0.0
        for _ in range(1000):  # Adjust based on needs
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rewards += float(reward)
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"{self.__class__.__name__} Validation Reward: {total_rewards}")
