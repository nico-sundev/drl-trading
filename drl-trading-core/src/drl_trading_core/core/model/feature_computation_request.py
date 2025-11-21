from typing import List
from attr import dataclass
from pandas import DataFrame

from drl_trading_core.core.model.dataset_identifier import DatasetIdentifier
from drl_trading_core.core.model.feature_definition import FeatureDefinition

@dataclass(frozen=True)
class FeatureComputationRequest:
    dataset_id: DatasetIdentifier
    feature_definitions: List[FeatureDefinition]
    market_data: DataFrame
