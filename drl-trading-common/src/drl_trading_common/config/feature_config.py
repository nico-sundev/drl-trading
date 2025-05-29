from typing import Any, Dict, List

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


class FeatureDefinition(BaseSchema):
    name: str
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[Dict[str, Any]]  # raw input from JSON
    parsed_parameter_sets: List[BaseParameterSetConfig] = Field(default_factory=list)


class FeaturesConfig(BaseSchema):
    feature_definitions: List[FeatureDefinition]
