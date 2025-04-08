from typing import List, Union
from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.base_schema import BaseSchema
from pydantic import model_validator
from ai_trading.config.feature_config_registry import FeatureConfigRegistry


class FeatureDefinition(BaseSchema):
    name: str
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[dict]  # raw input
    parsed_parameter_sets: List[BaseParameterSetConfig] = []  # becomes typed after validation

    @model_validator(mode="before")
    @classmethod
    def parse_parameter_sets(cls, data: dict) -> dict:
        name = data.get("name")
        raw_params = data.get("parameterSets", [])
        config_registry = FeatureConfigRegistry()

        config_cls = config_registry.feature_config_map.get(name.lower())
        if not config_cls:
            raise ValueError(f"No config class found for feature name '{name}'")

        # Inject type field dynamically before parsing as union
        for param in raw_params:
            param["type"] = name.lower()

        parsed_params = [config_cls(**p) for p in raw_params]
        data["parsed_parameter_sets"] = parsed_params
        return data


class FeaturesConfig(BaseSchema):
    feature_definitions: List[FeatureDefinition]
