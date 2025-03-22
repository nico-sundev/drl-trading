from typing import List

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
    def __init__(self, macd: MACDConfig, rsi_lengths: List[int], roc_lengths: List[int]):
        self.macd = macd
        self.rsi_lengths = rsi_lengths
        self.roc_lengths = roc_lengths

    def to_dict(self) -> dict:
        return {
            "macd": {
                "fast": self.macd.fast,
                "slow": self.macd.slow,
                "signal": self.macd.signal,
            },
            "rsi_lengths": self.rsi_lengths,
            "roc_lengths": self.roc_lengths,
        }