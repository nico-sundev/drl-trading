from abc import ABC, abstractmethod
from typing import List

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from ai_trading.common.policies.pol_grad_loss_cb import PolicyGradientLossCallback


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    Defines the common interface that all agent implementations must follow.
    """

    def __init__(
        self, env: VecEnv, total_timesteps: int, threshold: float = 0.5
    ) -> None:
        """
        Initialize and train a PPO agent.

        Args:
            env: Vectorized environment for training
            total_timesteps: Number of timesteps to train for
            threshold: Decision threshold for action recommendations
        """
        self.callback = PolicyGradientLossCallback()
        self.threshold = threshold
        self.env = env
        self.total_timesteps = total_timesteps

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate a prediction (action) based on the observation.

        Args:
            obs: Observation from the environment

        Returns:
            Action as numpy array
        """
        pass

    def action_to_recommendation(self, action: np.ndarray) -> List[str]:
        """
        Convert numeric action values to trading recommendations.

        Args:
            action: Numeric action values

        Returns:
            List of recommendations ('buy', 'sell', or 'hold')
        """
        recommendations = []
        for a in action:
            if a > self.threshold:
                recommendations.append("buy")
            elif a < -self.threshold:
                recommendations.append("sell")
            else:
                recommendations.append("hold")
        return recommendations

    @abstractmethod
    def validate(self, env: VecEnv) -> None:
        """
        Validate the agent's performance on the given environment.

        Args:
            env: Gymnasium environment to validate against
        """
        pass
