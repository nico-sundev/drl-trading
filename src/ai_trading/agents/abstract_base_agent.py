from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from gymnasium import Env


class AbstractBaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    Defines the common interface that all agent implementations must follow.
    """

    threshold: float = 0.5  # Default threshold value, can be overridden by subclasses

    @abstractmethod
    def predict(self, obs: Any) -> np.ndarray:
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
    def validate(self, env: Env) -> None:
        """
        Validate the agent's performance on the given environment.

        Args:
            env: Gymnasium environment to validate against
        """
        pass
