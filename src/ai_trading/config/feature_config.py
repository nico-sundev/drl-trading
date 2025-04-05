from typing import List, Union
from ai_trading.config.base_schema import BaseSchema
from pydantic import BaseModel, model_validator
from ai_trading.config.feature_config_mapper import (
    FEATURE_CONFIG_MAP,
    MacdConfig,
    RangeConfig,
    RocConfig,
    RsiConfig,
)


class FeatureDefinition(BaseSchema):
    name: str
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[dict]  # raw input
    parsed_parameter_sets: List[Union[
        MacdConfig, 
        RsiConfig,
        RocConfig,
        RangeConfig
        ]] = []

    @model_validator(mode="before")
    @classmethod
    def parse_parameter_sets(cls, data: dict) -> dict:
        name = data.get("name")
        raw_params = data.get("parameterSets", [])

        config_cls = FEATURE_CONFIG_MAP.get(name.lower())
        if not config_cls:
            raise ValueError(f"No config class found for feature name '{name}'")

        parsed_params = [config_cls(**p) for p in raw_params]
        data["parsed_parameter_sets"] = parsed_params
        return data


class FeaturesConfig(BaseSchema):
    feature_definitions: List[FeatureDefinition]
