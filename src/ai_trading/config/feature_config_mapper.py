from typing import Dict, Type
from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.base_schema import BaseSchema
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
  
class MacdConfig(BaseParameterSetConfig):
    fast: int
    slow: int
    signal: int
    
class RsiConfig(BaseParameterSetConfig):
    length: int
    
class RocConfig(BaseParameterSetConfig):
    length: int
    
class RangeConfig(BaseParameterSetConfig):
    lookback: int
    wick_handle_strategy: WICK_HANDLE_STRATEGY
    

FEATURE_CONFIG_MAP: Dict[str, Type[BaseSchema]] = {
    "macd": MacdConfig,
    "rsi": RsiConfig,
    "roc": RocConfig,
    "range": RangeConfig
}