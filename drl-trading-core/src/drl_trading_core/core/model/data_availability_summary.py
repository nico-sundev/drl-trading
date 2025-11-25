"""
Market data availability summary model.

This module defines the DataAvailabilitySummary model used for reporting
data availability across symbols and timeframes.
"""

from datetime import datetime
from typing import Optional

from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.core.model.timeframe import Timeframe


class DataAvailabilitySummary(BaseSchema):
    """
    Model representing data availability for a symbol-timeframe combination.

    Used for system monitoring and data discovery purposes.
    """

    symbol: str
    timeframe: Timeframe
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    record_count: int

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"DataAvailability(symbol={self.symbol}, timeframe={self.timeframe}, "
            f"records={self.record_count}, span={self.earliest_timestamp} to {self.latest_timestamp})"
        )

    @property
    def data_span_days(self) -> int:
        """Calculate the number of days covered by the data."""
        if not self.earliest_timestamp or not self.latest_timestamp:
            return 0
        return (self.latest_timestamp - self.earliest_timestamp).days

    @property
    def is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if the latest data is recent (within specified hours)."""
        if not self.latest_timestamp:
            return False
        from datetime import datetime, timedelta
        return self.latest_timestamp >= datetime.now() - timedelta(hours=max_age_hours)
