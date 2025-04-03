from typing import List

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY


class RangeConfig:

    def __init__(self, lookback: int = 500, wick_handle_strategy: WICK_HANDLE_STRATEGY = WICK_HANDLE_STRATEGY.PREVIOUS_WICK_ONLY):
        self.lookback = lookback
        self.wick_handle_strategy = wick_handle_strategy

    @property
    def lookback(self) -> int:
        return self._lookback

    @lookback.setter
    def lookback(self, value: int):
        if value <= 0:
            raise ValueError("Lookback must be positive")
        self._lookback = value

    @property
    def wick_handle_strategy(self) -> WICK_HANDLE_STRATEGY:
        return self._wick_handle_strategy

    @wick_handle_strategy.setter
    def wick_handle_strategy(self, value: WICK_HANDLE_STRATEGY):
        self._wick_handle_strategy = value


class MACDConfig:
    def __init__(self, fast: int, slow: int, signal: int):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def fast(self) -> int:
        return self._fast

    @fast.setter
    def fast(self, value: int):
        if value <= 0:
            raise ValueError("Fast MA length must be positive")
        self._fast = value

    @property
    def slow(self) -> int:
        return self._slow

    @slow.setter
    def slow(self, value: int):
        if value <= 0:
            raise ValueError("Slow MA length must be positive")
        self._slow = value

    @property
    def signal(self) -> int:
        return self._signal

    @signal.setter
    def signal(self, value: int):
        if value <= 0:
            raise ValueError("Signal MA length must be positive")
        self._signal = value


class FeatureConfig:
    def __init__(
        self,
        macd: MACDConfig,
        rsi_lengths: List[int],
        roc_lengths: List[int],
        range: RangeConfig,
    ):
        self.macd = macd
        self.rsi_lengths = rsi_lengths
        self.roc_lengths = roc_lengths
        self.range = range
