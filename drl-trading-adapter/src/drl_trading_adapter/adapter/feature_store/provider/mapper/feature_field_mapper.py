"""Feature field mapper for converting BaseFeature instances to Feast Field objects."""

import logging
from abc import ABC, abstractmethod

from feast import Field
from feast.types import Float32

from drl_trading_core.core.model.feature.feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


class IFeatureFieldFactory(ABC):
    """Interface for mapping FeatureMetadata to Feast Field objects."""

    @abstractmethod
    def create_fields(self, feature_metadata: FeatureMetadata) -> list[Field]:
        """
        Create Feast fields for the feature based on its type and role.

        Args:
            feature_metadata: The feature metadata for which fields are created

        Returns:
            list[Field]: List of Feast fields for the feature
        """
        pass


class FeatureFieldFactory(IFeatureFieldFactory):

    def create_fields(self, feature_metadata: FeatureMetadata) -> list[Field]:
        """
        Create fields for the feature view based on the feature's type and role.
        Current schema looks like:
        [field_base_name][_[sub_feature_name]] if sub-features exist

        Args:
            feature_metadata: The feature metadata for which fields are created

        Returns:
            list[Field]: List of fields for the feature view
        """
        feature_name = feature_metadata.__str__()
        logger.debug(f"Feast fields will be created for feature: {feature_name}")

        if len(feature_metadata.sub_feature_names) == 0:
            # If no sub-features, create a single field for the feature
            logger.debug(f"Creating feast field: {feature_name}")
            return [Field(name=feature_name, dtype=Float32)]

        fields = []
        for sub_feature in feature_metadata.sub_feature_names:
            # Combine feature name with sub-feature name to create unique field names
            feast_field_name = f"{feature_name}_{sub_feature}"
            logger.debug(f"Creating feast field: {feast_field_name}")
            fields.append(Field(name=feast_field_name, dtype=Float32))

        return fields
