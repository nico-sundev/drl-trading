from typing import Any, List

import numpy as np
from gymnasium import Env

from ai_trading.agents.abstract_base_agent import AbstractBaseAgent


class EnsembleAgent(AbstractBaseAgent):
    """
    Implementation of an ensemble agent that combines multiple agent predictions.
    This agent aggregates predictions from multiple base agents to make more robust decisions.
    """

    def __init__(
        self, env: Env, agents: List[AbstractBaseAgent], threshold: float = 0.5
    ) -> None:
        """
        Initialize the Ensemble agent.

        Args:
            env: Environment the agent will operate in
            agents: List of trained agent instances to ensemble
            threshold: Threshold for action recommendations
        """
        self.env = env
        self.agents = agents
        self.threshold = threshold

    def predict(self, obs: Any) -> Any:
        """
        Generate a prediction by aggregating predictions from all member agents.

        Args:
            obs: Observation from the environment

        Returns:
            Action as numpy array (aggregated from member agents)
        """
        predictions = [agent.predict(obs) for agent in self.agents]
        # Aggregate predictions (using mean as default strategy)
        return np.mean(predictions, axis=0)

    def validate(self, env: Env) -> None:
        """
        Validate the agent's performance on the given environment.

        Args:
            env: Gymnasium environment to validate against
        """
        obs, _ = env.reset()
        total_rewards = 0.0
        for _ in range(1000):  # Adjust based on needs
            action = self.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rewards += float(reward)
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"Ensemble Agent Validation Reward: {total_rewards}")
