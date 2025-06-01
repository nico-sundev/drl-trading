

from typing import Any, Callable, Dict

from drl_trading_common.interfaces.indicator_backend_registry_interface import (
    IndicatorBackendRegistryInterface,
)
from talipp.indicators import EMA, MACD, RSI


class IndicatorBackendRegistry(IndicatorBackendRegistryInterface):
    def __init__(self, mapping: Dict[str, Callable[..., Any]]):
        self.mapping = mapping

    def get_indicator(self, key: str, **kwargs) -> Any:
        if key not in self.mapping:
            raise ValueError(f"No indicator registered for '{key}'")
        return self.mapping[key](**kwargs)



def talipp_registry() -> IndicatorBackendRegistry:
    return IndicatorBackendRegistry({
        "rsi": lambda period: RSI(period=period),
        "ema": lambda period: EMA(period=period),
        "macd": lambda short, long_, signal: MACD(short_period=short, long_period=long_, signal_period=signal)
    })
