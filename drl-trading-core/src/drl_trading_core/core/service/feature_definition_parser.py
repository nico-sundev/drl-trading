from typing import List

from injector import inject

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)


@inject
class FeatureDefinitionParser:

    def __init__(self, feature_factory: IFeatureFactory) -> None:
        """
        Initialize the FeatureManagerService.

        Args:
            feature_factory: Factory for creating feature instances.
        """
        self.feature_factory = feature_factory

    def parse_feature_definition(self, feature_definition: FeatureDefinition) -> None:

        if not feature_definition.name:
            raise ValueError("Feature name is required")

        # early exit if parameter sets were already parsed
        if not feature_definition.parsed_parameter_sets:
            feature_definition.parsed_parameter_sets = {}

        for param_dict in feature_definition.parameter_sets:
            if not isinstance(param_dict, dict):
                raise ValueError(
                    f"Invalid parameter set: Expected a dictionary but got {type(param_dict).__name__}"
                )

            config_instance = self.feature_factory.create_config_instance(
                feature_definition.name, param_dict
            )

            # Only add non-None instances and avoid duplicates based on hash_id
            if (
                config_instance
                and feature_definition.parsed_parameter_sets.get(
                    config_instance.hash_id()
                )
                is None
            ):
                feature_definition.parsed_parameter_sets[config_instance.hash_id()] = (
                    config_instance
                )

        if (
            not feature_definition.parsed_parameter_sets
            and feature_definition.parameter_sets
        ):
            raise ValueError(
                f"Failed to parse any parameter sets for feature '{feature_definition.name}'"
            )

    def parse_feature_definitions(
        self, feature_definitions: List[FeatureDefinition]
    ) -> None:
        """
        Parse all feature definitions' parameter sets using the provided feature factory.

        Args:
            feature_factory: Factory for creating configuration instances
        """
        for feature_def in feature_definitions:
            self.parse_feature_definition(feature_def)
