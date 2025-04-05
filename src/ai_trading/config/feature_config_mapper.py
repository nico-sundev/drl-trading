

# Map feature names to config classes
from typing import Dict, Type

from ai_trading.config.base_schema import BaseSchema
from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
  
class MacdConfig(BaseSchema):
    fast: int
    slow: int
    signal: int
    
class RsiConfig(BaseSchema):
    length: int
    
class RocConfig(BaseSchema):
    length: int
    
class RangeConfig(BaseSchema):
    lookback: int
    wick_handle_strategy: WICK_HANDLE_STRATEGY
    

FEATURE_CONFIG_MAP: Dict[str, Type[BaseSchema]] = {
    "macd": MacdConfig,
    "rsi": RsiConfig,
    "roc": RocConfig,
    "range": RangeConfig
}