"""
Resampling response data model.

Contains the response container for completed resampling operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel


@dataclass(frozen=True)
class ResamplingResponse:
    """
    Response container for completed resampling operations.

    Contains the results of resampling operation including all generated
    higher timeframe data and metadata about the operation.
    """

    symbol: str
    base_timeframe: Timeframe
    resampled_data: Dict[Timeframe, List[MarketDataModel]]
    new_candles_count: Dict[Timeframe, int]
    processing_start_time: datetime
    processing_end_time: datetime
    source_records_processed: int

    @property
    def processing_duration_ms(self) -> int:
        """Get processing duration in milliseconds."""
        delta = self.processing_end_time - self.processing_start_time
        return int(delta.total_seconds() * 1000)

    @property
    def total_new_candles(self) -> int:
        """Get total number of newly generated candles across all timeframes."""
        return sum(self.new_candles_count.values())
