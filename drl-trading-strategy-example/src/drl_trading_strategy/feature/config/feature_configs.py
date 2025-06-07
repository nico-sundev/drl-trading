from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum


class RsiConfig(BaseParameterSetConfig):
    length: int

    @staticmethod
    def get_feature_type() -> FeatureTypeEnum:
        """Get the feature type enum for registration purposes.

        Returns:
            FeatureTypeEnum.RSI: The RSI feature type
        """
        return FeatureTypeEnum.RSI
