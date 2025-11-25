"""Unit tests for DataAvailabilityMapper.

Tests the mapping functionality for data availability query results,
including error handling and edge cases.
"""

import pytest
from datetime import datetime, timezone
from drl_trading_adapter.adapter.database.mapper.market_data_mapper import DataAvailabilityMapper
from drl_trading_core.core.model.data_availability_summary import DataAvailabilitySummary


class TestDataAvailabilityMapper:
    """Test suite for DataAvailabilityMapper.query_result_to_model() method."""

    def test_query_result_to_model_success(self) -> None:
        """Test successful conversion from query result to model."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        record_count = 1000
        earliest_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        latest_timestamp = datetime(2024, 1, 31, 23, 0, 0, tzinfo=timezone.utc)

        # When
        result = DataAvailabilityMapper.query_result_to_model(
            symbol=symbol,
            timeframe=timeframe,
            record_count=record_count,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
        )

        # Then
        assert isinstance(result, DataAvailabilitySummary)
        assert result.symbol == symbol
        assert result.timeframe.value == timeframe
        assert result.record_count == record_count
        assert result.earliest_timestamp == earliest_timestamp
        assert result.latest_timestamp == latest_timestamp

    def test_query_result_to_model_with_zero_records(self) -> None:
        """Test conversion when no records exist."""
        # Given
        symbol = "EMPTY"
        timeframe = "5m"
        record_count = 0
        earliest_timestamp = None
        latest_timestamp = None

        # When
        result = DataAvailabilityMapper.query_result_to_model(
            symbol=symbol,
            timeframe=timeframe,
            record_count=record_count,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
        )

        # Then
        assert result.symbol == symbol
        assert result.timeframe.value == timeframe
        assert result.record_count == 0
        assert result.earliest_timestamp is None
        assert result.latest_timestamp is None

    @pytest.mark.parametrize("symbol,timeframe,record_count", [
        ("TSLA", "1m", 50000),
        ("BTC-USD", "1d", 365),
        ("ETH", "4h", 2190),
        ("LONG_SYMBOL_NAME", "15m", 1)  # Edge case: long symbol
    ])
    def test_query_result_to_model_edge_cases(self, symbol: str, timeframe: str, record_count: int) -> None:
        """Test query_result_to_model with various edge case values."""
        # Given
        earliest_timestamp = datetime(2024, 1, 1) if record_count > 0 else None
        latest_timestamp = datetime(2024, 12, 31) if record_count > 0 else None

        # When
        result = DataAvailabilityMapper.query_result_to_model(
            symbol=symbol,
            timeframe=timeframe,
            record_count=record_count,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
        )

        # Then
        assert result.symbol == symbol
        assert result.timeframe.value == timeframe
        assert result.record_count == record_count

    def test_query_result_to_model_with_invalid_timeframe_raises_error(self) -> None:
        """Test query_result_to_model with invalid timeframe raises appropriate error."""
        # Given
        symbol = "INVALID"
        timeframe = "invalid_timeframe"
        record_count = 100
        earliest_timestamp = datetime(2024, 1, 1)
        latest_timestamp = datetime(2024, 1, 2)

        # When & Then
        with pytest.raises(ValueError, match="Failed to create DataAvailabilitySummary"):
            DataAvailabilityMapper.query_result_to_model(
                symbol=symbol,
                timeframe=timeframe,
                record_count=record_count,
                earliest_timestamp=earliest_timestamp,
                latest_timestamp=latest_timestamp
            )

    def test_query_result_to_model_with_negative_record_count_raises_error(self) -> None:
        """Test query_result_to_model with negative record count raises appropriate error."""
        # Given
        symbol = "TEST"
        timeframe = "1h"
        record_count = -1
        earliest_timestamp = datetime(2024, 1, 1)
        latest_timestamp = datetime(2024, 1, 2)

        # When & Then
        with pytest.raises(ValueError, match="Record count cannot be negative"):
            DataAvailabilityMapper.query_result_to_model(
                symbol=symbol,
                timeframe=timeframe,
                record_count=record_count,
                earliest_timestamp=earliest_timestamp,
                latest_timestamp=latest_timestamp
            )

    def test_query_result_to_model_with_inconsistent_timestamps(self) -> None:
        """Test query_result_to_model with inconsistent timestamp logic."""
        # Given - records exist but no timestamps provided
        symbol = "INCONSISTENT"
        timeframe = "1h"
        record_count = 100
        earliest_timestamp = None
        latest_timestamp = None

        # When
        result = DataAvailabilityMapper.query_result_to_model(
            symbol=symbol,
            timeframe=timeframe,
            record_count=record_count,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
        )

        # Then - should still create valid object even with inconsistent data
        assert result.symbol == symbol
        assert result.record_count == record_count
        assert result.earliest_timestamp is None
        assert result.latest_timestamp is None

    def test_query_result_to_model_with_same_timestamps(self) -> None:
        """Test query_result_to_model when earliest and latest timestamps are identical."""
        # Given
        symbol = "SINGLE"
        timeframe = "1d"
        record_count = 1
        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        # When
        result = DataAvailabilityMapper.query_result_to_model(
            symbol=symbol,
            timeframe=timeframe,
            record_count=record_count,
            earliest_timestamp=timestamp,
            latest_timestamp=timestamp
        )

        # Then
        assert result.earliest_timestamp == timestamp
        assert result.latest_timestamp == timestamp
        assert result.record_count == 1

    def test_query_result_to_model_with_empty_symbol_raises_error(self) -> None:
        """Test query_result_to_model with empty symbol raises appropriate error."""
        # Given
        symbol = ""
        timeframe = "1h"
        record_count = 0
        earliest_timestamp = None
        latest_timestamp = None

        # When & Then
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            DataAvailabilityMapper.query_result_to_model(
                symbol=symbol,
                timeframe=timeframe,
                record_count=record_count,
                earliest_timestamp=earliest_timestamp,
                latest_timestamp=latest_timestamp
            )
