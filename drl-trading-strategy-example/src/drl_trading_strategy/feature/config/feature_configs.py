from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_strategy.decorator.feature_type_decorator import feature_type
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum


@feature_type(FeatureTypeEnum.RSI)
class RsiConfig(BaseParameterSetConfig):
    length: int

@feature_type(FeatureTypeEnum.CLOSE_PRICE)
class ClosePriceConfig(BaseParameterSetConfig):
    """
    An empty configuration class that can be used as a placeholder.
    This is useful when no specific configuration is needed for a feature.
    """
    pass
