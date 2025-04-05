
from typing import List
from ai_trading.config.base_schema import BaseSchema

class RlModelConfig(BaseSchema):
    agents: List[str]
    training_split_ratio: float
    validating_split_ratio: float
    testing_split_ratio: float