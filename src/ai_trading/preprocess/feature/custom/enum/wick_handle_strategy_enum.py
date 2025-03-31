from enum import Enum

class WICK_HANDLE_STRATEGY(Enum):
    LAST_WICK_ONLY = "Last_wick_only"
    PREVIOUS_WICK_ONLY = "previous_wick_only"
    MEAN = "MEAN"
    MAX_BELOW_ATR = "MAX_BELOW_ATR"
