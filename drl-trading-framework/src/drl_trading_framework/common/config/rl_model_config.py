from typing import List

from drl_trading_framework.common.config.base_schema import BaseSchema


class RlModelConfig(BaseSchema):
    agents: List[str]  # Map agent names to their types
    training_split_ratio: float
    validating_split_ratio: float
    testing_split_ratio: float
    agent_threshold: float
    total_timesteps: int
