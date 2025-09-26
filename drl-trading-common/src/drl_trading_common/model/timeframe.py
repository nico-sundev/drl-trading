from enum import Enum


class Timeframe(Enum):
    """
    Enum representing different timeframes for trading data.
    """
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    def __str__(self):
        return self.value

    def to_minutes(self) -> int:
        """Convert timeframe to minutes for comparison and calculations."""
        timeframe_minutes = {
            Timeframe.MINUTE_1: 1,
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.MINUTE_30: 30,
            Timeframe.HOUR_1: 60,
            Timeframe.HOUR_4: 240,
            Timeframe.DAY_1: 1440,
            Timeframe.WEEK_1: 10080,
            Timeframe.MONTH_1: 43200,  # Approximate 30 days
            Timeframe.TICK: 0,  # Special case for tick data
        }
        return timeframe_minutes.get(self, 0)

    def to_seconds(self) -> int:
        """Convert timeframe to seconds for time difference calculations."""
        return self.to_minutes() * 60
