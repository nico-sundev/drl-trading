from dataclasses import dataclass
from typing import Optional

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""

    config: Optional[BaseParameterSetConfig]
    dataset_id: DatasetIdentifier
    feature_role: FeatureRoleEnum
    feature_name: str
    sub_feature_names: list[str]
    config_to_string: Optional[str] = None

    def __str__(self) -> str:
        """
        Create a string representation of the feature metadata.

        Returns a unique identifier in the format:
        - With config: "[feature_name]_[config_to_string]_[config_hash]"
        - Without config: "[feature_name]"

        Returns:
            str: String representation of the feature metadata
        """
        config_string = (
            f"_{self.config_to_string}_{self.config.hash_id()}" if self.config else ""
        )
        return f"{self.feature_name}{config_string}"
