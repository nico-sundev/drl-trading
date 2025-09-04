"""Feature field mapper for converting BaseFeature instances to Feast Field objects."""

import logging
from abc import ABC, abstractmethod

from feast import Field
from feast.types import Float32

from drl_trading_common.base import BaseFeature

logger = logging.getLogger(__name__)


class IFeatureFieldMapper(ABC):
    """Interface for mapping BaseFeature instances to Feast Field objects."""

    @abstractmethod
    def get_field_base_name(self, feature: BaseFeature) -> str:
        """
        Create a unique field name based on the feature name and its config hash.

        Args:
            feature: The feature object

        Returns:
            str: A unique name for the field
        """
        pass

    @abstractmethod
    def create_fields(self, feature: BaseFeature) -> list[Field]:
        """
        Create Feast fields for the feature based on its type and role.

        Args:
            feature: The feature for which fields are created

        Returns:
            list[Field]: List of Feast fields for the feature
        """
        pass


class FeatureFieldMapper(IFeatureFieldMapper):
    """
    Maps BaseFeature instances to Feast Field objects.

    This class encapsulates the logic for:
    - Generating unique field names from feature metadata
    - Creating Feast Field objects with proper data types
    - Handling sub-features and feature hierarchies

    Separating this logic allows for:
    - Better testing (can mock the mapper in integration tests)
    - Single Responsibility Principle adherence
    - Easier maintenance and modification of mapping logic
    """

    def get_field_base_name(self, feature: BaseFeature) -> str:
        """
        Create a unique field name based on the feature name and its config hash.
        Current schema looks like:
        [feature_name]_[config_to_string]_[config_hash]

        Example 1: A feature relying on a config
        If feature name is "rsi", config_to_string is "14" and config_hash is "abc123",
        the resulting name will be "rsi_14_abc123".

        Example 2: A feature without a config
        If feature name is "close_price",
        the resulting name will be "close_price".

        Args:
            feature: The feature object

        Returns:
            str: A unique name for the field
        """
        config = feature.get_config()
        config_string = (
            f"_{feature.get_config_to_string()}_{config.hash_id()}" if config else ""
        )
        return f"{feature.get_feature_name()}{config_string}"

    def create_fields(self, feature: BaseFeature) -> list[Field]:
        """
        Create fields for the feature view based on the feature's type and role.
        Current schema looks like:
        [field_base_name][_[sub_feature_name]] if sub-features exist

        Args:
            feature: The feature for which fields are created

        Returns:
            list[Field]: List of fields for the feature view
        """
        feature_name = self.get_field_base_name(feature)
        logger.debug(f"Feast fields will be created for feature: {feature_name}")

        if len(feature.get_sub_features_names()) == 0:
            # If no sub-features, create a single field for the feature
            logger.debug(f"Creating feast field: {feature_name}")
            return [Field(name=feature_name, dtype=Float32)]

        fields = []
        for sub_feature in feature.get_sub_features_names():
            # Combine feature name with sub-feature name to create unique field names
            feast_field_name = f"{feature_name}_{sub_feature}"
            logger.debug(f"Creating feast field: {feast_field_name}")
            fields.append(Field(name=feast_field_name, dtype=Float32))

        return fields
