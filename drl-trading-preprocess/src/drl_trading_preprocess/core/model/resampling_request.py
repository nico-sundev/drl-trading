"""
Resampling request data model.

Contains the request container for resampling operations with validation logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from drl_trading_common.model.timeframe import Timeframe


@dataclass(frozen=True)
class ResamplingRequest:
    """
    Request container for resampling operations.

    Encapsulates all parameters needed to perform OHLCV resampling
    from a base timeframe to multiple target timeframes.
    """

    symbol: str
    base_timeframe: Timeframe
    target_timeframes: List[Timeframe]
    start_from_timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Symbol cannot be empty")

        if not self.target_timeframes:
            raise ValueError("Target timeframes cannot be empty")

        # Validate that target timeframes are larger than base timeframe
        base_minutes = self.base_timeframe.to_minutes()
        for target_tf in self.target_timeframes:
            target_minutes = target_tf.to_minutes()
            if target_minutes <= base_minutes:
                raise ValueError(
                    f"Target timeframe {target_tf.value} must be larger than "
                    f"base timeframe {self.base_timeframe.value}"
                )
