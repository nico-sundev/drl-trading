from typing import List

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_strategy.feature.feature_factory import (
    IFeatureFactory,
)


def parse_parameters(
    feature_definition: FeatureDefinition, feature_factory: IFeatureFactory
) -> None:
    """
    Parse raw parameter sets using the provided feature factory.

    This method converts the raw parameter dictionaries into properly typed
    configuration objects using the appropriate config class for this feature.

    Args:
        feature_factory: Factory for creating configuration instances

    Raises:
        ValueError: If no config class is found for this feature, or if
                   parameter parsing fails
    """
    if not feature_definition.name:
        raise ValueError("Feature name is required")

    # early exit if parameter sets were already parsed
    if feature_definition.parsed_parameter_sets:
        return

    feature_definition.parsed_parameter_sets = []

    for param_dict in feature_definition.parameter_sets:
        if not isinstance(param_dict, dict):
            raise ValueError(
                f"Invalid parameter set: Expected a dictionary but got {type(param_dict).__name__}"
            )

        config_instance = feature_factory.create_config_instance(
            feature_definition.name, param_dict
        )

        if config_instance:
            feature_definition.parsed_parameter_sets.append(config_instance)

    if (
        not feature_definition.parsed_parameter_sets
        and feature_definition.parameter_sets
    ):
        raise ValueError(
            f"Failed to parse any parameter sets for feature '{feature_definition.name}'"
        )

def parse_all_parameters(
    feature_definitions: List[FeatureDefinition],
    feature_factory: IFeatureFactory,
) -> None:
    """
    Parse all feature definitions' parameter sets using the provided feature factory.

    Args:
        feature_factory: Factory for creating configuration instances
    """
    for feature_def in feature_definitions:
        parse_parameters(feature_def, feature_factory)
