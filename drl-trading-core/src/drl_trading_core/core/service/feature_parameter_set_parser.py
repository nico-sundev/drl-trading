from typing import List

from injector import inject

from drl_trading_core.core.service.feature.feature_factory_interface import (
    IFeatureFactory,
)
from drl_trading_core.core.model.feature_definition import FeatureDefinition


@inject
class FeatureParameterSetParser:

    def __init__(self, feature_factory: IFeatureFactory) -> None:
        """
        Initialize the FeatureManagerService.

        Args:
            feature_factory: Factory for creating feature instances.
        """
        self.feature_factory = feature_factory

    def parse_parameter_set(
        self, feature_definition: FeatureDefinition, raw_parameter_set: dict
    ) -> None:

        config_instance = self.feature_factory.create_config_instance(
            feature_definition.name, raw_parameter_set
        )

        # Only add non-None instances and avoid duplicates based on hash_id
        if (
            config_instance
            and feature_definition.parsed_parameter_sets.get(config_instance.hash_id())
            is None
        ):
            feature_definition.parsed_parameter_sets[config_instance.hash_id()] = (
                config_instance
            )

    def parse_feature_definitions(
        self,
        feature_definitions: List[FeatureDefinition],
    ) -> None:
        """
        Parse all feature definitions' parameter sets using the provided feature factory.

        Args:
            feature_factory: Factory for creating configuration instances
        """
        for feature_def in feature_definitions:
            for param_dict in feature_def.raw_parameter_sets:
                if not feature_def.name:
                    raise ValueError("Feature name is required")

                # early exit if parameter sets were already parsed
                # if not feature_def.parsed_parameter_sets:
                #     feature_def.parsed_parameter_sets = {}

                if not isinstance(param_dict, dict):
                    raise ValueError(
                        f"Invalid parameter set: Expected a dictionary but got {type(param_dict).__name__}"
                    )

                self.parse_parameter_set(
                    feature_def, param_dict
                )
