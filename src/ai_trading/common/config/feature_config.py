from typing import Any, Dict, List

from pydantic import Field

from ai_trading.common.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.common.config.base_schema import BaseSchema
from ai_trading.common.config.feature_config_factory import (
    FeatureConfigFactoryInterface,
)


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

    def parse_parameters(self, config_factory: FeatureConfigFactoryInterface) -> None:
        """
        Parse raw parameter sets using the provided config factory.

        This method converts the raw parameter dictionaries into properly typed
        configuration objects using the appropriate config class for this feature.

        Args:
            config_factory: Factory for creating configuration instances

        Raises:
            ValueError: If no config class is found for this feature, or if
                       parameter parsing fails
        """
        if not self.name:
            raise ValueError("Feature name is required")

        # Clear any existing parsed parameters
        self.parsed_parameter_sets = []

        for param_dict in self.parameter_sets:
            if not isinstance(param_dict, dict):
                raise ValueError(
                    f"Invalid parameter set: Expected a dictionary but got {type(param_dict).__name__}"
                )

            config_instance = config_factory.create_config_instance(
                self.name, param_dict
            )

            if config_instance:
                self.parsed_parameter_sets.append(config_instance)

        if not self.parsed_parameter_sets and self.parameter_sets:
            raise ValueError(
                f"Failed to parse any parameter sets for feature '{self.name}'"
            )


class FeaturesConfig(BaseSchema):
    feature_definitions: List[FeatureDefinition]

    def parse_all_parameters(
        self, config_factory: FeatureConfigFactoryInterface
    ) -> None:
        """
        Parse all feature definitions' parameter sets using the provided config factory.

        Args:
            config_factory: Factory for creating configuration instances
        """
        for feature_def in self.feature_definitions:
            feature_def.parse_parameters(config_factory)
