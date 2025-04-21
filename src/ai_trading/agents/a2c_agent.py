from gymnasium import Env
from stable_baselines3 import A2C

from ai_trading.agents.agent_policy import AgentPolicy


class A2CAgent(AgentPolicy[A2C]):
    """
    Implementation of an agent using Advantage Actor-Critic (A2C).
    """

    def __init__(self, env: Env, total_timesteps: int, threshold: float = 0.5) -> None:
        """
        Initialize the A2C agent.

        Args:
            env: Training environment
            total_timesteps: Number of timesteps for training
            threshold: Threshold for action recommendations
        """
        super().__init__(env, total_timesteps, threshold)
