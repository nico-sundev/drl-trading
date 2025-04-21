from gymnasium import Env
from stable_baselines3 import SAC

from ai_trading.agents.agent_policy import AgentPolicy


class SACAgent(AgentPolicy[SAC]):
    """
    Implementation of an agent using Soft Actor-Critic (SAC).
    """

    def __init__(self, env: Env, total_timesteps: int, threshold: float = 0.5) -> None:
        """
        Initialize the SAC agent.

        Args:
            env: Training environment
            total_timesteps: Number of timesteps for training
            threshold: Threshold for action recommendations
        """
        super().__init__(env, total_timesteps, threshold)
