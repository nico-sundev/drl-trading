import importlib
import inspect
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
    
class RviConfig(BaseParameterSetConfig):
    length: int
    
class RangeConfig(BaseParameterSetConfig):
    lookback: int
    wick_handle_strategy: WICK_HANDLE_STRATEGY
    

FEATURE_CONFIG_MAP: Dict[str, Type[BaseSchema]] = {
    "macd": MacdConfig,
    "rsi": RsiConfig,
    "roc": RocConfig,
    "range": RangeConfig,
    "rvi": RviConfig
}
def discover_config_classes() -> Dict[str, Type[BaseParameterSetConfig]]:
    config_map = {}
    config_module = importlib.import_module("ai_trading.config")

    for name, obj in inspect.getmembers(config_module, inspect.isclass):
        if issubclass(obj, BaseParameterSetConfig) and obj is not BaseParameterSetConfig:
            feature_name = name.replace("Config", "").lower()
            config_map[feature_name] = obj

    return config_map