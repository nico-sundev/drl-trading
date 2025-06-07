
from dataclasses import dataclass


@dataclass
class RSIIndicatorResult:
    """
    Represents the result of a Relative Strength Index (RSI) calculation.

    Attributes:
        rsi (list[float]): The calculated RSI values.
    """
    rsi: list[float]
