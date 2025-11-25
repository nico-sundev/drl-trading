from dataclasses import dataclass
from typing import Optional

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""

    config: Optional[BaseParameterSetConfig]
    dataset_id: DatasetIdentifier
    feature_role: FeatureRoleEnum
    feature_name: str
    sub_feature_names: list[str]
    config_to_string: Optional[str] = None
