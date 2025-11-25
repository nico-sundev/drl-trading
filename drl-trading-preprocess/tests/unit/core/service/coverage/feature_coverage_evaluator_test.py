"""
Unit tests for FeatureCoverageEvaluator.

Tests the business logic for evaluating feature coverage analysis results,
ensuring proper separation of concerns from the data models.
"""
import pytest
from datetime import datetime, timezone, timedelta

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
    FeatureCoverageAnalysis,
    FeatureCoverageInfo,
    ComputationPeriod,
    WarmupPeriod,
)
from drl_trading_preprocess.core.service.coverage.feature_coverage_evaluator import (
    FeatureCoverageEvaluator,
)


class TestFeatureCoverageEvaluator:
    """Test suite for FeatureCoverageEvaluator service."""

    @pytest.fixture
    def evaluator(self) -> FeatureCoverageEvaluator:
        """Create evaluator instance for testing."""
        return FeatureCoverageEvaluator()

    @pytest.fixture
    def sample_coverage_analysis(self) -> FeatureCoverageAnalysis:
        """Create a sample coverage analysis for testing."""
        return FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                ),
                "feature_b": FeatureCoverageInfo(
                    feature_name="feature_b",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
                    record_count=144,
                    coverage_percentage=50.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 12, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
                "feature_c": FeatureCoverageInfo(
                    feature_name="feature_c",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

    def test_get_fully_covered_features(self, evaluator: FeatureCoverageEvaluator, sample_coverage_analysis: FeatureCoverageAnalysis) -> None:
        """Test getting fully covered features."""
        result = evaluator.get_fully_covered_features(sample_coverage_analysis)
        assert result == ["feature_a"]

    def test_get_partially_covered_features(self, evaluator: FeatureCoverageEvaluator, sample_coverage_analysis: FeatureCoverageAnalysis) -> None:
        """Test getting partially covered features."""
        result = evaluator.get_partially_covered_features(sample_coverage_analysis)
        assert result == ["feature_b"]

    def test_get_missing_features(self, evaluator: FeatureCoverageEvaluator, sample_coverage_analysis: FeatureCoverageAnalysis) -> None:
        """Test getting missing features."""
        result = evaluator.get_missing_features(sample_coverage_analysis)
        assert result == ["feature_c"]

    def test_get_features_needing_computation_all_missing(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test getting features needing computation when all are missing."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_features_needing_computation(analysis)
        assert result == ["feature_a"]

    def test_get_features_needing_computation_partial_coverage(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test getting features needing computation with partial coverage."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                ),
                "feature_b": FeatureCoverageInfo(
                    feature_name="feature_b",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
                    record_count=144,
                    coverage_percentage=50.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 12, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_features_needing_computation(analysis)
        assert result == ["feature_b"]

    def test_get_features_needing_warmup(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test getting features needing warmup."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                ),
                "feature_b": FeatureCoverageInfo(
                    feature_name="feature_b",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_features_needing_warmup(analysis)
        assert result == ["feature_b"]

    def test_get_overall_coverage_percentage(self, evaluator: FeatureCoverageEvaluator, sample_coverage_analysis: FeatureCoverageAnalysis) -> None:
        """Test calculating overall coverage percentage."""
        result = evaluator.get_overall_coverage_percentage(sample_coverage_analysis)
        # (100 + 50 + 0) / 3 = 50.0
        assert result == 50.0

    def test_get_overall_coverage_percentage_empty_features(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test calculating overall coverage percentage when no features exist."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={},  # Empty feature coverage
            existing_features_df=None,
        )

        result = evaluator.get_overall_coverage_percentage(analysis)
        assert result == 0.0

    def test_is_ohlcv_constrained_true(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test OHLCV constraint detection when constrained."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),  # Starts 6 hours late
            ohlcv_latest_timestamp=datetime(2023, 1, 1, 18, tzinfo=timezone.utc),  # Ends 6 hours early
            ohlcv_record_count=144,  # Only half the requested period
            adjusted_start_time=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 1, 18, tzinfo=timezone.utc),
            feature_coverage={},
            existing_features_df=None,
        )

        result = evaluator.is_ohlcv_constrained(analysis)
        assert result is True

    def test_is_ohlcv_constrained_false(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test OHLCV constraint detection when not constrained."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={},
            existing_features_df=None,
        )

        result = evaluator.is_ohlcv_constrained(analysis)
        assert result is False

    def test_get_computation_period_no_missing_features(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test computation period when no features need computation."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_computation_period(analysis)
        expected = ComputationPeriod(
            start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
        )
        assert result == expected

    def test_get_computation_period_with_missing_features(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test computation period when features need computation."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),  # This is what get_computation_period returns
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
                    record_count=144,
                    coverage_percentage=50.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 12, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_computation_period(analysis)
        expected = ComputationPeriod(
            start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),  # Returns adjusted_start_time
            end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
        )
        assert result == expected

    def test_get_warmup_period_no_warmup_needed(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when no warmup is needed."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)
        assert result is None

    def test_get_warmup_period_with_warmup_needed(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period calculation when warmup is needed."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),  # Start after existing data
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),  # Has some data but missing later period
                    record_count=72,  # Has some records
                    coverage_percentage=25.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 6, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=10)  # Small warmup for test
        # For 5-minute timeframe, 10 candles = 50 minutes before the latest timestamp
        expected_start = datetime(2023, 1, 1, 6, tzinfo=timezone.utc) - timedelta(minutes=50)
        expected = WarmupPeriod(
            start_time=expected_start,
            end_time=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),
        )
        assert result == expected

    def test_get_warmup_period_no_existing_records_for_warmup_features(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when features needing warmup have no existing records."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,  # No existing records
                    coverage_percentage=0.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)
        assert result is None

    def test_get_warmup_period_insufficient_ohlcv_data(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when total available OHLCV candles is less than warmup requirement."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 1, 2, tzinfo=timezone.utc),  # Only 2 hours = 24 candles
            ohlcv_record_count=24,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 2, tzinfo=timezone.utc),
                    record_count=24,
                    coverage_percentage=100.0,  # Fully covered for available period
                    missing_periods=[
                        (datetime(2023, 1, 1, 2, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)  # Requires 500 candles, only 24 available
        assert result is None

    def test_get_warmup_period_no_ohlcv_timestamps(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when OHLCV timestamps are None."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=False,
            ohlcv_earliest_timestamp=None,
            ohlcv_latest_timestamp=None,
            ohlcv_record_count=0,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),
                    record_count=72,
                    coverage_percentage=25.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 6, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)
        assert result is None

    def test_get_warmup_period_no_latest_timestamps(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when no latest timestamps are available for warmup features."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,  # No latest timestamp
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)
        assert result is None

    def test_get_warmup_period_computation_starts_within_covered_period(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when computation period starts within the covered period."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 3, tzinfo=timezone.utc),  # Starts within covered period
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),  # Covers up to 6 AM
                    record_count=72,
                    coverage_percentage=25.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 6, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=500)
        assert result is None  # No warmup needed since computation starts within covered period

    def test_get_warmup_period_constrained_by_ohlcv_availability(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when warmup start needs to be constrained by OHLCV availability."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, 2, tzinfo=timezone.utc),  # OHLCV starts at 2 AM
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),
                    record_count=72,
                    coverage_percentage=25.0,
                    missing_periods=[
                        (datetime(2023, 1, 1, 6, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=100)  # Large warmup that would go before OHLCV start
        # Warmup should be constrained to start at OHLCV earliest timestamp
        expected = WarmupPeriod(
            start_time=datetime(2023, 1, 1, 2, tzinfo=timezone.utc),  # Constrained by OHLCV availability
            end_time=datetime(2023, 1, 1, 6, tzinfo=timezone.utc),
        )
        assert result == expected

    def test_get_warmup_period_warmup_start_after_warmup_end(self, evaluator: FeatureCoverageEvaluator) -> None:
        """Test warmup period when calculated warmup start is after warmup end."""
        analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=288,
            adjusted_start_time=datetime(2023, 1, 1, 12, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            feature_coverage={
                "feature_a": FeatureCoverageInfo(
                    feature_name="feature_a",
                    is_fully_covered=False,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 1, 0, 1, tzinfo=timezone.utc),  # Very close to start
                    record_count=1,
                    coverage_percentage=0.3,
                    missing_periods=[
                        (datetime(2023, 1, 1, 0, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))
                    ],
                ),
            },
            existing_features_df=None,
        )

        result = evaluator.get_warmup_period(analysis, warmup_candles=10000)  # Very large warmup
        assert result is None  # Warmup start would be before warmup end, so no warmup

    def test_get_summary_message(self, evaluator: FeatureCoverageEvaluator, sample_coverage_analysis: FeatureCoverageAnalysis) -> None:
        """Test generating summary message."""
        result = evaluator.get_summary_message(sample_coverage_analysis)
        expected = (
            "Feature coverage for BTCUSD 5m: 1/3 fully covered (50.0%), 1 missing, 1 partial. "
            "OHLCV: 288 records [2023-01-01 00:00:00+00:00 - 2023-01-02 00:00:00+00:00]"
        )
        assert result == expected
