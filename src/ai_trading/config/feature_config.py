from typing import Any, Dict, List

from pydantic import model_validator

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.base_schema import BaseSchema
from ai_trading.config.feature_config_registry import FeatureConfigRegistry


class FeatureStoreConfig(BaseSchema):
    enabled: bool = False
    repo_path: str = "feature_repo"
    offline_store_path: str = "data/features.parquet"
    entity_name: str = "symbol"
    ttl_days: int = 365
    online_enabled: bool = True


class FeatureDefinition(BaseSchema):
    name: str
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[Dict[str, Any]]  # raw input
    parsed_parameter_sets: List[BaseParameterSetConfig] = (
        []
    )  # becomes typed after validation

    @model_validator(mode="before")
    @classmethod
    def parse_parameter_sets(cls, data: dict) -> dict:
        name = data.get("name")
        if not name:
            raise ValueError("Feature name is required")

        raw_params = data.get("parameterSets", [])
        config_registry = FeatureConfigRegistry()

        config_cls = config_registry.feature_config_map.get(name.lower())
        if not config_cls:
            raise ValueError(f"No config class found for feature name '{name}'")

        # Inject type field dynamically before parsing as union
        for param in raw_params:
            if isinstance(param, dict):
                param["type"] = name.lower()

        parsed_params = [config_cls(**p) for p in raw_params if isinstance(p, dict)]
        data["parsed_parameter_sets"] = parsed_params
        return data


class FeaturesConfig(BaseSchema):
    feature_definitions: List[FeatureDefinition]
