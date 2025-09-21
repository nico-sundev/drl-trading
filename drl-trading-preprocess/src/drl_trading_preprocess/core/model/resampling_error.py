"""
Resampling error data model.

Contains error information for failed resampling operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from drl_trading_common.model.timeframe import Timeframe


@dataclass(frozen=True)
class ResamplingError:
    """
    Error information for failed resampling operations.

    Provides structured error reporting for resampling failures
    with context and debugging information.
    """

    symbol: str
    base_timeframe: Timeframe
    target_timeframes: List[Timeframe]
    error_message: str
    error_type: str
    timestamp: datetime
    context: Dict[str, str]

    def to_dict(self) -> Dict[str, str]:
        """Convert error to dictionary for logging/messaging."""
        return {
            "symbol": self.symbol,
            "base_timeframe": self.base_timeframe.value,
            "target_timeframes": ",".join([tf.value for tf in self.target_timeframes]),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat(),
            **self.context
        }
