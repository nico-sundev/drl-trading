"""
Test cases for the Timeframe enum model.

Tests all methods of the Timeframe enum including conversion utilities
for minutes, seconds, and pandas frequency strings.
"""
import pytest
import pandas as pd
from datetime import datetime

from drl_trading_common.core.model.timeframe import Timeframe


class TestTimeframe:
    """Test cases for Timeframe enum conversion methods."""

    def test_to_minutes_conversion(self) -> None:
        """Test that timeframes convert correctly to minutes."""
        # Given/When/Then
        assert Timeframe.MINUTE_1.to_minutes() == 1
        assert Timeframe.MINUTE_5.to_minutes() == 5
        assert Timeframe.MINUTE_15.to_minutes() == 15
        assert Timeframe.MINUTE_30.to_minutes() == 30
        assert Timeframe.HOUR_1.to_minutes() == 60
        assert Timeframe.HOUR_4.to_minutes() == 240
        assert Timeframe.DAY_1.to_minutes() == 1440
        assert Timeframe.WEEK_1.to_minutes() == 10080
        assert Timeframe.MONTH_1.to_minutes() == 43200  # Approximate 30 days
        assert Timeframe.TICK.to_minutes() == 0

    def test_to_seconds_conversion(self) -> None:
        """Test that timeframes convert correctly to seconds."""
        # Given/When/Then
        assert Timeframe.MINUTE_1.to_seconds() == 60
        assert Timeframe.MINUTE_5.to_seconds() == 300
        assert Timeframe.MINUTE_15.to_seconds() == 900
        assert Timeframe.MINUTE_30.to_seconds() == 1800
        assert Timeframe.HOUR_1.to_seconds() == 3600
        assert Timeframe.HOUR_4.to_seconds() == 14400
        assert Timeframe.DAY_1.to_seconds() == 86400
        assert Timeframe.WEEK_1.to_seconds() == 604800
        assert Timeframe.MONTH_1.to_seconds() == 2592000
        assert Timeframe.TICK.to_seconds() == 0

    def test_to_pandas_freq_valid_timeframes(self) -> None:
        """Test that valid timeframes convert correctly to pandas frequency strings."""
        # Given/When/Then
        assert Timeframe.MINUTE_1.to_pandas_freq() == "1min"
        assert Timeframe.MINUTE_5.to_pandas_freq() == "5min"
        assert Timeframe.MINUTE_15.to_pandas_freq() == "15min"
        assert Timeframe.MINUTE_30.to_pandas_freq() == "30min"
        assert Timeframe.HOUR_1.to_pandas_freq() == "1h"
        assert Timeframe.HOUR_4.to_pandas_freq() == "4h"
        assert Timeframe.DAY_1.to_pandas_freq() == "1D"
        assert Timeframe.WEEK_1.to_pandas_freq() == "1W"
        assert Timeframe.MONTH_1.to_pandas_freq() == "1MS"

    def test_to_pandas_freq_tick_raises_error(self) -> None:
        """Test that TICK timeframe raises ValueError when converting to pandas frequency."""
        # Given/When/Then
        with pytest.raises(ValueError, match="TICK timeframe cannot be converted to pandas frequency"):
            Timeframe.TICK.to_pandas_freq()

    @pytest.mark.parametrize(
        "timeframe,expected_freq",
        [
            (Timeframe.MINUTE_1, "1min"),
            (Timeframe.MINUTE_5, "5min"),
            (Timeframe.MINUTE_15, "15min"),
            (Timeframe.MINUTE_30, "30min"),
            (Timeframe.HOUR_1, "1h"),
            (Timeframe.HOUR_4, "4h"),
            (Timeframe.DAY_1, "1D"),
            (Timeframe.WEEK_1, "1W"),
            (Timeframe.MONTH_1, "1MS"),
        ]
    )
    def test_pandas_frequency_compatibility(self, timeframe: Timeframe, expected_freq: str) -> None:
        """Test that pandas frequency strings are valid and can be used with pd.date_range."""
        # Given - use different time ranges for different frequencies
        if timeframe in [Timeframe.WEEK_1, Timeframe.MONTH_1]:
            # For weekly and monthly, use a longer range
            start_time = datetime(2024, 1, 1)
            end_time = datetime(2024, 2, 1)
        else:
            # For shorter frequencies, use a 2-hour window
            start_time = datetime(2024, 1, 1, 10, 0, 0)
            end_time = datetime(2024, 1, 1, 12, 0, 0)

        # When
        pandas_freq = timeframe.to_pandas_freq()

        # Then - verify it matches expected frequency
        assert pandas_freq == expected_freq

        # Then - verify it works with pandas date_range (basic smoke test)
        try:
            date_range = pd.date_range(start=start_time, end=end_time, freq=pandas_freq)
            # For weekly/monthly frequencies, we might get 0 results in short ranges, which is valid
            assert len(date_range) >= 0  # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Pandas frequency {pandas_freq} is not valid: {e}")

    def test_string_representation(self) -> None:
        """Test that timeframes have correct string representations."""
        # Given/When/Then
        assert str(Timeframe.MINUTE_1) == "1m"
        assert str(Timeframe.MINUTE_5) == "5m"
        assert str(Timeframe.MINUTE_15) == "15m"
        assert str(Timeframe.MINUTE_30) == "30m"
        assert str(Timeframe.HOUR_1) == "1h"
        assert str(Timeframe.HOUR_4) == "4h"
        assert str(Timeframe.DAY_1) == "1d"
        assert str(Timeframe.WEEK_1) == "1w"
        assert str(Timeframe.MONTH_1) == "1M"
        assert str(Timeframe.TICK) == "tick"

    def test_enum_value_consistency(self) -> None:
        """Test that enum values are consistent."""
        # Given/When/Then
        assert Timeframe.MINUTE_1.value == "1m"
        assert Timeframe.MINUTE_5.value == "5m"
        assert Timeframe.MINUTE_15.value == "15m"
        assert Timeframe.MINUTE_30.value == "30m"
        assert Timeframe.HOUR_1.value == "1h"
        assert Timeframe.HOUR_4.value == "4h"
        assert Timeframe.DAY_1.value == "1d"
        assert Timeframe.WEEK_1.value == "1w"
        assert Timeframe.MONTH_1.value == "1M"
        assert Timeframe.TICK.value == "tick"


class TestTimeframePandasIntegration:
    """Test pandas integration scenarios to ensure compatibility."""

    def test_date_range_generation_for_common_timeframes(self) -> None:
        """Test that pandas date range generation works for common use cases."""
        # Given
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)  # 1 hour range

        test_cases = [
            (Timeframe.MINUTE_1, 61),   # 60 minutes + 1 for inclusive end
            (Timeframe.MINUTE_5, 13),   # 12 intervals + 1 for inclusive end
            (Timeframe.MINUTE_15, 5),   # 4 intervals + 1 for inclusive end
            (Timeframe.MINUTE_30, 3),   # 2 intervals + 1 for inclusive end
            (Timeframe.HOUR_1, 2),      # 1 interval + 1 for inclusive end
        ]

        for timeframe, expected_count in test_cases:
            # When
            pandas_freq = timeframe.to_pandas_freq()
            date_range = pd.date_range(start=start_time, end=end_time, freq=pandas_freq)

            # Then
            assert len(date_range) == expected_count, f"Expected {expected_count} intervals for {timeframe}, got {len(date_range)}"

    def test_monthly_frequency_behavior(self) -> None:
        """Test that monthly frequency uses month start (MS) correctly."""
        # Given
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 4, 1)  # 3 months

        # When
        pandas_freq = Timeframe.MONTH_1.to_pandas_freq()
        date_range = pd.date_range(start=start_time, end=end_time, freq=pandas_freq)

        # Then
        assert pandas_freq == "1MS"  # Month start frequency
        assert len(date_range) == 4  # Jan 1, Feb 1, Mar 1, Apr 1
        assert date_range[0].day == 1  # Should start at beginning of month
        assert date_range[1].day == 1  # Should align to month starts

    def test_weekly_frequency_behavior(self) -> None:
        """Test that weekly frequency works correctly."""
        # Given - start on a Sunday to align with pandas weekly frequency default
        start_time = datetime(2023, 12, 31)  # Sunday
        end_time = datetime(2024, 1, 21)     # ~3 weeks later

        # When
        pandas_freq = Timeframe.WEEK_1.to_pandas_freq()
        date_range = pd.date_range(start=start_time, end=end_time, freq=pandas_freq)

        # Then
        assert pandas_freq == "1W"
        assert len(date_range) >= 2  # Should have at least 2 weekly intervals
        # Weekly intervals should be 7 days apart
        for i in range(1, len(date_range)):
            time_diff = date_range[i] - date_range[i-1]
            assert time_diff.days == 7
