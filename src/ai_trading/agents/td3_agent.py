from gymnasium import Env
from stable_baselines3 import TD3

from ai_trading.agents.agent_policy import AgentPolicy


class TD3Agent(AgentPolicy[TD3]):
    """
    Implementation of an agent using Twin Delayed DDPG (TD3).
    """

    def __init__(self, env: Env, total_timesteps: int, threshold: float = 0.5) -> None:
        """
        Initialize the TD3 agent.

        Args:
            env: Training environment
            total_timesteps: Number of timesteps for training
            threshold: Threshold for action recommendations
        """
        super().__init__(env, total_timesteps, threshold)
