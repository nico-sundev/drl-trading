"""
Unit tests for FeatureCoverageAnalyzer.

Tests the analyzer's ability to:
- Check OHLCV data availability
- Batch fetch features from Feast
- Analyze coverage for individual features
- Handle edge cases (no data, partial coverage, full coverage)
- Identify missing periods/gaps in feature data
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
import pandas as pd

from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.data_availability_summary import DataAvailabilitySummary
from drl_trading_preprocess.core.service.coverage.feature_coverage_analyzer import FeatureCoverageAnalyzer
from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    FeatureCoverageAnalysis
)


# Constants for test data
TEST_SYMBOL = "EURUSD"
TEST_TIMEFRAME = Timeframe.HOUR_1
START_TIME = datetime(2024, 1, 1, 0, 0, 0)
END_TIME = datetime(2024, 1, 1, 23, 0, 0)
FEATURE_NAMES = ["rsi", "sma", "ema"]


class TestFeatureCoverageAnalyzerInitialization:
    """Test analyzer initialization."""

    def test_initialization_success(self) -> None:
        """Test successful initialization with dependencies."""
        # Given
        mock_feature_store = Mock()
        mock_market_data_reader = Mock()

        # When
        analyzer = FeatureCoverageAnalyzer(
            feature_store_fetch_port=mock_feature_store,
            market_data_reader=mock_market_data_reader
        )

        # Then
        assert analyzer.feature_store_fetch_port == mock_feature_store
        assert analyzer.market_data_reader == mock_market_data_reader


class TestOHLCVAvailabilityChecking:
    """Test OHLCV data availability checking."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=Mock(),
            market_data_reader=Mock()
        )

    def test_check_ohlcv_availability_with_data(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test OHLCV availability check when data exists."""
        # Given
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=100,
            earliest_timestamp=START_TIME,
            latest_timestamp=END_TIME
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)

        # When
        result = analyzer._check_ohlcv_availability(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME
        )

        # Then
        assert result["available"] is True
        assert result["record_count"] == 100
        assert result["earliest_timestamp"] == START_TIME
        assert result["latest_timestamp"] == END_TIME
        analyzer.market_data_reader.get_data_availability.assert_called_once_with(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME
        )

    def test_check_ohlcv_availability_no_data(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test OHLCV availability check when no data exists."""
        # Given
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=0,
            earliest_timestamp=None,
            latest_timestamp=None
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)

        # When
        result = analyzer._check_ohlcv_availability(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME
        )

        # Then
        assert result["available"] is False
        assert result["record_count"] == 0
        assert result["earliest_timestamp"] is None
        assert result["latest_timestamp"] is None

    def test_check_ohlcv_availability_no_overlap(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test OHLCV availability check when requested period doesn't overlap with available data."""
        # Given
        # Available data: Jan 1-10
        # Requested: Jan 20-30 (no overlap)
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=100,
            earliest_timestamp=datetime(2024, 1, 1),
            latest_timestamp=datetime(2024, 1, 10)
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)
        requested_start = datetime(2024, 1, 20)
        requested_end = datetime(2024, 1, 30)

        # When
        result = analyzer._check_ohlcv_availability(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=requested_start,
            end_time=requested_end
        )

        # Then
        assert result["available"] is False
        assert result["record_count"] == 0
        # Should still return the actual data bounds
        assert result["earliest_timestamp"] == datetime(2024, 1, 1)
        assert result["latest_timestamp"] == datetime(2024, 1, 10)

    def test_check_ohlcv_availability_exception_handling(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test OHLCV availability check handles exceptions gracefully."""
        # Given
        analyzer.market_data_reader.get_data_availability = Mock(
            side_effect=Exception("Database connection error")
        )

        # When
        result = analyzer._check_ohlcv_availability(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME
        )

        # Then
        assert result["available"] is False
        assert result["record_count"] == 0
        assert result["earliest_timestamp"] is None
        assert result["latest_timestamp"] is None


class TestBatchFetchFeatures:
    """Test batch feature fetching from Feast."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=Mock(),
            market_data_reader=Mock()
        )

    @pytest.fixture
    def feature_config(self) -> FeatureConfigVersionInfo:
        """Create feature configuration."""
        return FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="abc123",
            created_at=datetime(2024, 1, 1),
            feature_definitions=[]
        )

    def test_batch_fetch_features_success(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test successful batch fetch of features."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        mock_df = pd.DataFrame({
            'rsi': [50.0] * len(timestamps),
            'sma': [1.2] * len(timestamps),
            'ema': [1.3] * len(timestamps)
        }, index=timestamps)
        analyzer.feature_store_fetch_port.get_offline = Mock(return_value=mock_df)

        # When
        result = analyzer._batch_fetch_features(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME,
            feature_config_version_info=feature_config
        )

        # Then
        assert not result.empty
        assert len(result) == len(timestamps)
        assert 'rsi' in result.columns
        assert 'sma' in result.columns
        assert 'ema' in result.columns
        analyzer.feature_store_fetch_port.get_offline.assert_called_once()

    def test_batch_fetch_features_empty_result(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test batch fetch when no features exist."""
        # Given
        analyzer.feature_store_fetch_port.get_offline = Mock(return_value=pd.DataFrame())

        # When
        result = analyzer._batch_fetch_features(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME,
            feature_config_version_info=feature_config
        )

        # Then
        assert result.empty

    def test_batch_fetch_features_no_timestamps(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test batch fetch when period generates no timestamps."""
        # Given
        # Invalid period that generates no timestamps
        invalid_start = datetime(2024, 1, 1, 0, 0, 0)
        invalid_end = datetime(2024, 1, 1, 0, 0, 0)  # Same time

        # When
        result = analyzer._batch_fetch_features(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=invalid_start,
            end_time=invalid_end,
            feature_config_version_info=feature_config
        )

        # Then
        assert result.empty

    def test_batch_fetch_features_exception_handling(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test batch fetch handles exceptions gracefully."""
        # Given
        analyzer.feature_store_fetch_port.get_offline = Mock(
            side_effect=Exception("Feast connection error")
        )

        # When
        result = analyzer._batch_fetch_features(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            start_time=START_TIME,
            end_time=END_TIME,
            feature_config_version_info=feature_config
        )

        # Then
        assert result.empty


class TestIndividualFeatureAnalysis:
    """Test individual feature coverage analysis."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=Mock(),
            market_data_reader=Mock()
        )

    def test_analyze_fully_covered_features(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test analysis of features with full coverage."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        features_df = pd.DataFrame({
            'rsi': [50.0] * len(timestamps),
            'sma': [1.2] * len(timestamps)
        }, index=timestamps)

        # When
        result = analyzer._analyze_individual_features(
            feature_names=['rsi', 'sma'],
            existing_features_df=features_df,
            adjusted_start_time=START_TIME,
            adjusted_end_time=END_TIME,
            timeframe=TEST_TIMEFRAME
        )

        # Then
        assert 'rsi' in result
        assert 'sma' in result
        assert result['rsi'].is_fully_covered is True
        assert result['sma'].is_fully_covered is True
        assert result['rsi'].coverage_percentage >= 99.0
        assert result['sma'].coverage_percentage >= 99.0
        assert len(result['rsi'].missing_periods) == 0
        assert len(result['sma'].missing_periods) == 0

    def test_analyze_partially_covered_features(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test analysis of features with partial coverage (gaps)."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        # Create feature with some null values (gaps)
        rsi_values = [50.0] * len(timestamps)
        rsi_values[5] = None  # Gap
        rsi_values[6] = None  # Gap
        rsi_values[15] = None  # Another gap

        features_df = pd.DataFrame({
            'rsi': rsi_values
        }, index=timestamps)

        # When
        result = analyzer._analyze_individual_features(
            feature_names=['rsi'],
            existing_features_df=features_df,
            adjusted_start_time=START_TIME,
            adjusted_end_time=END_TIME,
            timeframe=TEST_TIMEFRAME
        )

        # Then
        assert 'rsi' in result
        assert result['rsi'].is_fully_covered is False
        assert result['rsi'].coverage_percentage < 100.0
        assert result['rsi'].record_count == len(timestamps) - 3  # 3 nulls
        assert len(result['rsi'].missing_periods) > 0

    def test_analyze_missing_features(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test analysis of features that don't exist in Feast."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        features_df = pd.DataFrame({
            'rsi': [50.0] * len(timestamps)
        }, index=timestamps)

        # When
        result = analyzer._analyze_individual_features(
            feature_names=['rsi', 'missing_feature'],
            existing_features_df=features_df,
            adjusted_start_time=START_TIME,
            adjusted_end_time=END_TIME,
            timeframe=TEST_TIMEFRAME
        )

        # Then
        assert 'missing_feature' in result
        assert result['missing_feature'].is_fully_covered is False
        assert result['missing_feature'].coverage_percentage == 0.0
        assert result['missing_feature'].record_count == 0
        assert result['missing_feature'].earliest_timestamp is None
        assert result['missing_feature'].latest_timestamp is None
        assert len(result['missing_feature'].missing_periods) == 1

    def test_analyze_empty_dataframe(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test analysis when features DataFrame is empty."""
        # Given
        empty_df = pd.DataFrame()

        # When
        result = analyzer._analyze_individual_features(
            feature_names=['rsi', 'sma'],
            existing_features_df=empty_df,
            adjusted_start_time=START_TIME,
            adjusted_end_time=END_TIME,
            timeframe=TEST_TIMEFRAME
        )

        # Then
        assert 'rsi' in result
        assert 'sma' in result
        assert result['rsi'].coverage_percentage == 0.0
        assert result['sma'].coverage_percentage == 0.0
        assert result['rsi'].is_fully_covered is False
        assert result['sma'].is_fully_covered is False


class TestMissingPeriodsIdentification:
    """Test identification of missing periods in feature data."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=Mock(),
            market_data_reader=Mock()
        )

    def test_identify_no_missing_periods(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test identification when feature has full coverage."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        feature_series = pd.Series([50.0] * len(timestamps), index=timestamps)

        # When
        result = analyzer._identify_missing_periods(
            feature_series=feature_series,
            expected_timestamps=timestamps
        )

        # Then
        assert len(result) == 0

    def test_identify_single_missing_period(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test identification of a single consecutive missing period."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        values = [50.0] * len(timestamps)
        # Create a gap from index 5 to 8
        values[5] = None
        values[6] = None
        values[7] = None
        values[8] = None
        feature_series = pd.Series(values, index=timestamps)

        # When
        result = analyzer._identify_missing_periods(
            feature_series=feature_series,
            expected_timestamps=timestamps
        )

        # Then
        assert len(result) >= 1
        # Should identify at least one missing period
        start, end = result[0]
        assert start == timestamps[5]
        assert end == timestamps[8]

    def test_identify_multiple_missing_periods(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test identification of multiple separate missing periods."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        values = [50.0] * len(timestamps)
        # Create two separate gaps
        values[5] = None
        values[6] = None
        values[15] = None
        values[16] = None
        feature_series = pd.Series(values, index=timestamps)

        # When
        result = analyzer._identify_missing_periods(
            feature_series=feature_series,
            expected_timestamps=timestamps
        )

        # Then
        assert len(result) >= 2

    def test_identify_missing_periods_at_boundaries(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test identification of missing periods at start/end boundaries."""
        # Given
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        values = [50.0] * len(timestamps)
        # Missing at start
        values[0] = None
        values[1] = None
        # Missing at end
        values[-2] = None
        values[-1] = None
        feature_series = pd.Series(values, index=timestamps)

        # When
        result = analyzer._identify_missing_periods(
            feature_series=feature_series,
            expected_timestamps=timestamps
        )

        # Then
        assert len(result) >= 2


class TestFullCoverageAnalysis:
    """Test complete coverage analysis workflow."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        mock_feature_store = Mock()
        mock_market_data_reader = Mock()
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=mock_feature_store,
            market_data_reader=mock_market_data_reader
        )

    @pytest.fixture
    def feature_config(self) -> FeatureConfigVersionInfo:
        """Create feature configuration."""
        return FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="abc123",
            created_at=datetime(2024, 1, 1),
            feature_definitions=[]
        )

    def test_analyze_coverage_success_with_data(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test successful coverage analysis with available data."""
        # Given
        # Mock OHLCV availability
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=24,
            earliest_timestamp=START_TIME,
            latest_timestamp=END_TIME
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)

        # Mock feature data
        timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq='1h')
        mock_df = pd.DataFrame({
            'rsi': [50.0] * len(timestamps),
            'sma': [1.2] * len(timestamps)
        }, index=timestamps)
        analyzer.feature_store_fetch_port.get_offline = Mock(return_value=mock_df)

        # When
        result = analyzer.analyze_feature_coverage(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            base_timeframe=Timeframe.MINUTE_1,
            feature_names=['rsi', 'sma'],
            requested_start_time=START_TIME,
            requested_end_time=END_TIME,
            feature_config_version_info=feature_config
        )

        # Then
        assert isinstance(result, FeatureCoverageAnalysis)
        assert result.symbol == TEST_SYMBOL
        assert result.timeframe == TEST_TIMEFRAME
        assert result.ohlcv_available is True
        assert len(result.feature_coverage) == 2
        assert 'rsi' in result.feature_coverage
        assert 'sma' in result.feature_coverage
        assert result.feature_coverage['rsi'].is_fully_covered is True
        assert result.feature_coverage['sma'].is_fully_covered is True

    def test_analyze_coverage_no_ohlcv_data(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test coverage analysis when no OHLCV data exists."""
        # Given
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=0,
            earliest_timestamp=None,
            latest_timestamp=None
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)

        # When
        result = analyzer.analyze_feature_coverage(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            base_timeframe=Timeframe.MINUTE_1,
            feature_names=['rsi', 'sma'],
            requested_start_time=START_TIME,
            requested_end_time=END_TIME,
            feature_config_version_info=feature_config
        )

        # Then
        assert isinstance(result, FeatureCoverageAnalysis)
        assert result.ohlcv_available is False
        assert result.ohlcv_record_count == 0
        assert len(result.feature_coverage) == 2
        # All features should be marked as missing
        assert result.feature_coverage['rsi'].coverage_percentage == 0.0
        assert result.feature_coverage['sma'].coverage_percentage == 0.0

    def test_analyze_coverage_adjusted_time_period(
        self, analyzer: FeatureCoverageAnalyzer, feature_config: FeatureConfigVersionInfo
    ) -> None:
        """Test coverage analysis with time period adjusted by OHLCV availability."""
        # Given
        # OHLCV data available from Jan 1 00:00 to Jan 1 12:00 (12 hours)
        # But user requests Jan 1 00:00 to Jan 1 23:00 (24 hours)
        ohlcv_end = datetime(2024, 1, 1, 12, 0, 0)
        mock_availability = DataAvailabilitySummary(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            record_count=13,
            earliest_timestamp=START_TIME,
            latest_timestamp=ohlcv_end
        )
        analyzer.market_data_reader.get_data_availability = Mock(return_value=mock_availability)

        timestamps = pd.date_range(start=START_TIME, end=ohlcv_end, freq='1h')
        mock_df = pd.DataFrame({
            'rsi': [50.0] * len(timestamps)
        }, index=timestamps)
        analyzer.feature_store_fetch_port.get_offline = Mock(return_value=mock_df)

        # When
        result = analyzer.analyze_feature_coverage(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            base_timeframe=Timeframe.MINUTE_1,
            feature_names=['rsi'],
            requested_start_time=START_TIME,
            requested_end_time=END_TIME,  # Request until 23:00
            feature_config_version_info=feature_config
        )

        # Then
        assert result.requested_start_time == START_TIME
        assert result.requested_end_time == END_TIME
        # Adjusted time should be constrained to OHLCV availability
        assert result.adjusted_start_time == START_TIME
        assert result.adjusted_end_time == ohlcv_end


class TestNoDataAnalysisCreation:
    """Test creation of no-data analysis results."""

    @pytest.fixture
    def analyzer(self) -> FeatureCoverageAnalyzer:
        """Create analyzer with mocked dependencies."""
        return FeatureCoverageAnalyzer(
            feature_store_fetch_port=Mock(),
            market_data_reader=Mock()
        )

    def test_create_no_data_analysis(self, analyzer: FeatureCoverageAnalyzer) -> None:
        """Test creation of analysis result indicating no data."""
        # Given
        feature_names = ['rsi', 'sma', 'ema']

        # When
        result = analyzer._create_no_data_analysis(
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            feature_names=feature_names,
            requested_start_time=START_TIME,
            requested_end_time=END_TIME,
            adjusted_start_time=START_TIME,
            adjusted_end_time=END_TIME
        )

        # Then
        assert isinstance(result, FeatureCoverageAnalysis)
        assert result.ohlcv_available is False
        assert result.ohlcv_record_count == 0
        assert len(result.feature_coverage) == 3

        # All features should be missing
        for feature_name in feature_names:
            assert feature_name in result.feature_coverage
            coverage = result.feature_coverage[feature_name]
            assert coverage.is_fully_covered is False
            assert coverage.coverage_percentage == 0.0
            assert coverage.record_count == 0
            assert coverage.earliest_timestamp is None
            assert coverage.latest_timestamp is None
            assert len(coverage.missing_periods) == 1
