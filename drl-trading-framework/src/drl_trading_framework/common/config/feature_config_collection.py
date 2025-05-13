from typing import Literal

from drl_trading_framework.common.config.base_parameter_set_config import (
    BaseParameterSetConfig,
)
from drl_trading_framework.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


class MacdConfig(BaseParameterSetConfig):
    type: Literal["macd"]
    fast_length: int
    slow_length: int
    signal_length: int


class RsiConfig(BaseParameterSetConfig):
    type: Literal["rsi"]
    length: int


class RocConfig(BaseParameterSetConfig):
    type: Literal["roc"]
    length: int


class RviConfig(BaseParameterSetConfig):
    type: Literal["rvi"]
    length: int


class RangeConfig(BaseParameterSetConfig):
    type: Literal["range"]
    lookback: int
    wick_handle_strategy: WICK_HANDLE_STRATEGY


class BollbandsConfig(BaseParameterSetConfig):
    type: Literal["bollbands"]
    length: int
    std_dev: float = 2.0  # Adding std_dev with default value
