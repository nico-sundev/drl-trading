from typing import List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from ai_trading.agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    Implementation of an agent using Proximal Policy Optimization (PPO).

    This agent works with vectorized environments (VecEnv) from Stable Baselines3,
    which allow for efficient parallel environment execution.
    """

    def __init__(
        self, env: VecEnv, total_timesteps: int, threshold: float = 0.5
    ) -> None:
        super().__init__(env, total_timesteps, threshold)
        # Initialize and train the model
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate a prediction using the trained model.

        Args:
            obs: Observation from the environment

        Returns:
            Action as numpy array
        """
        action, _ = self.model.predict(np.array(obs), deterministic=True)
        return action

    def validate(self, env: VecEnv) -> None:
        """
        Validate the agent's performance on the given vectorized environment.

        Args:
            env: Vectorized environment to validate against
        """
        obs = env.reset()
        total_rewards = 0.0

        for _ in range(1000):  # Adjust based on needs
            action, _ = self.model.predict(np.array(obs), deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # Sum rewards across all environments
            total_rewards += float(np.sum(rewards))

            # No need to manually reset as VecEnv handles this automatically

        print(f"{self.__class__.__name__} Validation Reward: {total_rewards}")

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
