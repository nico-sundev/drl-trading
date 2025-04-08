import importlib
from typing import Literal
from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
  
class MacdConfig(BaseParameterSetConfig):
    type: Literal["macd"]
    fast: int
    slow: int
    signal: int
    
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