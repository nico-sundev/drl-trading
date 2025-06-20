from typing import Any, Dict, List

from drl_trading_common.model.timeframe import Timeframe
from pydantic import Field

from ..base.base_parameter_set_config import BaseParameterSetConfig
from ..base.base_schema import BaseSchema


class FeatureStoreConfig(BaseSchema):
    enabled: bool
    repo_path: str
    offline_store_path: str
    entity_name: str
    ttl_days: int
    online_enabled: bool
    service_name: str
    service_version: str

class FeatureDefinition(BaseSchema):
    """Feature definition configuration.

    Uses string-based names to avoid circular dependencies between common library
    and strategy-specific enums. The strategy layer can provide type-safe conversion
    to enums when needed.
    """
    name: str  # Feature type identifier (e.g., "rsi", "macd")
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[Dict[str, Any]]  # raw input from JSON
    parsed_parameter_sets: List[BaseParameterSetConfig] = Field(default_factory=list)

class FeaturesConfig(BaseSchema):
    dataset_definitions: Dict[str, List[Timeframe]]
    feature_definitions: List[FeatureDefinition]
