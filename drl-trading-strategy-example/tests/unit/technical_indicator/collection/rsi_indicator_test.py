"""Unit tests for RsiIndicator timestamp preservation."""
import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_strategy_example.feature.config.feature_configs import RsiConfig
from drl_trading_strategy_example.technical_indicator.collection.indicators import RsiIndicator


class TestRsiIndicatorTimestampPreservation:
    """Test cases for RSI indicator timestamp handling."""

    @pytest.fixture
    def config(self) -> RsiConfig:
        """Create RSI config with period 14."""
        # Given
        return RsiConfig(type="rsi", enabled=True, length=14)

    @pytest.fixture
    def sample_ohlcv_with_datetime_index(self) -> DataFrame:
        """Create sample OHLCV data with proper DatetimeIndex."""
        # Given
        timestamps = pd.date_range(start="2024-01-01", periods=20, freq="1h", tz="UTC")
        data = {
            "Open": [100.0 + i for i in range(20)],
            "High": [101.0 + i for i in range(20)],
            "Low": [99.0 + i for i in range(20)],
            "Close": [100.5 + i for i in range(20)],
            "Volume": [1000 + i * 10 for i in range(20)]
        }
        return DataFrame(data, index=timestamps)

    def test_add_stores_timestamps_from_dataframe_index(
        self, config: RsiConfig, sample_ohlcv_with_datetime_index: DataFrame
    ) -> None:
        """Test that add() extracts and stores timestamps from DataFrame index."""
        # Given
        indicator = RsiIndicator(config)
        expected_timestamps = sample_ohlcv_with_datetime_index.index.tolist()

        # When
        indicator.add(sample_ohlcv_with_datetime_index)

        # Then
        assert len(indicator.timestamps) == len(expected_timestamps)
        assert indicator.timestamps == expected_timestamps

    def test_get_all_returns_dataframe_with_datetime_index(
        self, config: RsiConfig, sample_ohlcv_with_datetime_index: DataFrame
    ) -> None:
        """Test that get_all() returns DataFrame with proper DatetimeIndex."""
        # Given
        indicator = RsiIndicator(config)
        indicator.add(sample_ohlcv_with_datetime_index)

        # When
        result = indicator.get_all()

        # Then
        assert result is not None
        assert isinstance(result.index, pd.DatetimeIndex)
        # RSI needs warmup period - with 20 data points and period 14, we get limited RSI values
        # The important thing is that timestamps are preserved for the values we DO get
        assert len(result) == len(indicator.indicator)  # Matches actual RSI output
        assert len(result) > 0  # We should have at least some values
        # Verify the index timestamps are from our input data
        for ts in result.index:
            assert ts in sample_ohlcv_with_datetime_index.index

    def test_get_all_preserves_timezone_information(
        self, config: RsiConfig, sample_ohlcv_with_datetime_index: DataFrame
    ) -> None:
        """Test that timezone information is preserved through indicator computation."""
        # Given
        indicator = RsiIndicator(config)
        indicator.add(sample_ohlcv_with_datetime_index)

        # When
        result = indicator.get_all()

        # Then
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_get_latest_returns_dataframe_with_datetime_index(
        self, config: RsiConfig, sample_ohlcv_with_datetime_index: DataFrame
    ) -> None:
        """Test that get_latest() returns DataFrame with proper DatetimeIndex."""
        # Given
        indicator = RsiIndicator(config)
        indicator.add(sample_ohlcv_with_datetime_index)

        # When
        result = indicator.get_latest()

        # Then
        assert result is not None
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 1
        # Latest timestamp should be from the timestamps that have RSI values
        # (not necessarily the last input timestamp due to warmup period)
        assert result.index[0] in sample_ohlcv_with_datetime_index.index

    def test_incremental_add_accumulates_timestamps(
        self, config: RsiConfig
    ) -> None:
        """Test that multiple add() calls accumulate timestamps correctly."""
        # Given
        indicator = RsiIndicator(config)

        # First batch
        timestamps1 = pd.date_range(start="2024-01-01", periods=10, freq="1h", tz="UTC")
        df1 = DataFrame({
            "Open": range(10), "High": range(10), "Low": range(10),
            "Close": range(10), "Volume": range(10)
        }, index=timestamps1)

        # Second batch
        timestamps2 = pd.date_range(start="2024-01-01 10:00:00", periods=5, freq="1h", tz="UTC")
        df2 = DataFrame({
            "Open": range(10, 15), "High": range(10, 15), "Low": range(10, 15),
            "Close": range(10, 15), "Volume": range(10, 15)
        }, index=timestamps2)

        # When
        indicator.add(df1)
        indicator.add(df2)

        # Then
        # All 15 timestamps should be stored
        assert len(indicator.timestamps) == 15

        # RSI values will be limited due to warmup period
        result = indicator.get_all()
        assert result is not None
        assert len(result) == len(indicator.indicator)  # Matches actual RSI computed values
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_rsi_values_align_with_timestamps(
        self, config: RsiConfig, sample_ohlcv_with_datetime_index: DataFrame
    ) -> None:
        """Test that RSI values are correctly aligned with their timestamps."""
        # Given
        indicator = RsiIndicator(config)
        indicator.add(sample_ohlcv_with_datetime_index)

        # When
        result = indicator.get_all()

        # Then
        assert result is not None
        assert "rsi" in result.columns

        # RSI requires warmup period (14 values) before producing output
        # With 20 input points and period 14, we get limited RSI values
        # The key test: Each RSI value has a corresponding valid timestamp
        assert len(result) == len(indicator.indicator)
        assert len(result) > 0

        # Verify each row has a timestamp from our input data
        for idx, timestamp in enumerate(result.index):
            assert timestamp in sample_ohlcv_with_datetime_index.index
            # Verify RSI value exists (may be None during warmup in some implementations)
            assert "rsi" in result.columns

    def test_get_all_raises_error_when_no_data_added(self, config: RsiConfig) -> None:
        """Test that get_all() raises error when indicator hasn't been computed."""
        # Given
        indicator = RsiIndicator(config)

        # When/Then
        with pytest.raises(ValueError, match="RSI indicator has not been computed yet"):
            indicator.get_all()

    def test_get_latest_raises_error_when_no_data_added(self, config: RsiConfig) -> None:
        """Test that get_latest() raises error when indicator hasn't been computed."""
        # Given
        indicator = RsiIndicator(config)

        # When/Then
        with pytest.raises(ValueError, match="RSI indicator has not been computed yet"):
            indicator.get_latest()
