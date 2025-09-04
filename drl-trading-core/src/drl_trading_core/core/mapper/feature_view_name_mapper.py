from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_core.core.enum.feature_view_name_enum import (
    FeatureViewNameEnum,
)


class FeatureViewNameMapper:
    """
    A class to map feature view names to their corresponding feature store names.
    """

    def map(self, feature_role: FeatureRoleEnum) -> str:
        """
        Maps a feature role to its corresponding feature view name.

        Args:
            feature_role (FeatureRoleEnum): The role of the feature.

        Returns:
            str: The name of the feature view.
        """
        if feature_role == FeatureRoleEnum.OBSERVATION_SPACE:
            return str(FeatureViewNameEnum.OBSERVATION_SPACE.value)
        elif feature_role == FeatureRoleEnum.REWARD_ENGINEERING:
            return str(FeatureViewNameEnum.REWARD_ENGINEERING.value)
        else:
            raise ValueError(f"Unknown feature role: {feature_role}")
