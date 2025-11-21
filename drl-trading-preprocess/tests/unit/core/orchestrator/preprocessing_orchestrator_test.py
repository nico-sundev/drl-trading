"""
Tests for PreprocessingOrchestrator fire-and-forget architecture.

Tests the orchestrator's coordination logic and message publishing behavior.
"""
from typing import TYPE_CHECKING

from drl_trading_common.model.timeframe import Timeframe

if TYPE_CHECKING:
    from unittest.mock import Mock
    from pandas import DataFrame
    from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator
    from builders import FeaturePreprocessingRequestBuilder


class TestPreprocessingOrchestratorSuccessFlow:
    """Test successful preprocessing with proper dependency mocking."""

    def test_successful_processing_publishes_completion_notification(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test end-to-end successful processing flow."""
        # Given
        request = request_builder.for_training().with_skip_existing(False).build()

        # Configure mock resampler to return data
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        # Configure validator to pass all features
        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        # Configure feature computer to return features
        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # Configure coverage analyzer to indicate resampling is needed
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=False,  # Forces resampling
            ohlcv_earliest_timestamp=None,
            ohlcv_latest_timestamp=None,
            ohlcv_record_count=0,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Since OHLCV not available
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,  # Needs computation
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[],
                )
            },
            existing_features_df=None,
        )
        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.return_value = coverage_analysis

        # Configure coverage evaluator to return features needing computation
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_computation.return_value = ['test_feature']
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_warmup.return_value = []

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None, "Fire-and-forget should return None"

        # Verify message publisher was called
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        # Verify call arguments
        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['request'] == request
        assert call_args.kwargs['processing_context'] == "training"
        assert call_args.kwargs['total_features_computed'] == 2
        assert call_args.kwargs['timeframes_processed'] == [Timeframe.MINUTE_5]

        # Verify success details contain metrics
        success_details = call_args.kwargs['success_details']
        assert 'features_5m' in success_details
        assert 'records_5m' in success_details

    def test_multiple_timeframes_publishes_correct_metrics(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test processing multiple timeframes publishes correct feature counts."""
        # Given
        request = (request_builder
                   .for_training()
                   .with_skip_existing(False)
                   .with_target_timeframes([Timeframe.MINUTE_5, Timeframe.MINUTE_15])
                   .build())

        # Configure mock resampler to return data for both timeframes
        resampling_response = resampling_response_factory(
            timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15]
        )
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        # Configure validator to pass all features
        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        # Configure feature computer to return features
        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # Configure coverage analyzer for both timeframes
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_analyses = {}
        for tf in [Timeframe.MINUTE_5, Timeframe.MINUTE_15]:
            coverage_analysis = FeatureCoverageAnalysis(
                symbol="BTCUSD",
                timeframe=tf,
                requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
                requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
                ohlcv_available=False,  # Forces resampling
                ohlcv_earliest_timestamp=None,
                ohlcv_latest_timestamp=None,
                ohlcv_record_count=0,
                adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
                adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
                requires_resampling=True,  # Since OHLCV not available
                feature_coverage={
                    'test_feature': FeatureCoverageInfo(
                        feature_name='test_feature',
                        is_fully_covered=False,  # Needs computation
                        earliest_timestamp=None,
                        latest_timestamp=None,
                        record_count=0,
                        coverage_percentage=0.0,
                        missing_periods=[],
                    )
                },
                existing_features_df=None,
            )
            coverage_analyses[tf] = coverage_analysis

        # Mock the analyze_feature_coverage to return different results for different timeframes
        def mock_analyze_coverage(symbol, timeframe, **kwargs):
            return coverage_analyses[timeframe]

        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.side_effect = mock_analyze_coverage

        # Configure coverage evaluator
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_computation.return_value = ['test_feature']
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_warmup.return_value = []

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None, "Fire-and-forget should return None"

        # Verify message publisher was called
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        # Verify call arguments
        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['request'] == request
        assert call_args.kwargs['processing_context'] == "training"
        assert call_args.kwargs['total_features_computed'] == 4  # 2 features × 2 timeframes
        timeframes_processed = call_args.kwargs['timeframes_processed']
        assert Timeframe.MINUTE_5 in timeframes_processed
        assert Timeframe.MINUTE_15 in timeframes_processed


class TestPreprocessingOrchestratorErrorHandling:
    """Test error handling and notification publishing."""

    def test_validation_errors_publish_failure_notification(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
    ) -> None:
        """Test that validation errors trigger failure notification."""
        # Given
        request = request_builder.for_training().with_skip_existing(False).build()

        # Configure validator to fail all features
        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': False
        }

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None, "Fire-and-forget should return None even on failure"

        # Verify feature validation error notification was published
        mock_dependencies['message_publisher'].publish_feature_validation_error.assert_called_once()

        # Verify error details contain validation info
        call_args = mock_dependencies['message_publisher'].publish_feature_validation_error.call_args
        assert call_args.kwargs['request'] == request
        assert 'invalid_features' in call_args.kwargs

    def test_storage_failure_publishes_failure_notification(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test that storage failures trigger failure notification."""
        # Given
        request = request_builder.for_training().with_skip_existing(False).build()

        # Configure resampler and validator to succeed
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # Configure coverage analyzer to indicate resampling is needed
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_analysis = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=False,  # Forces resampling
            ohlcv_earliest_timestamp=None,
            ohlcv_latest_timestamp=None,
            ohlcv_record_count=0,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Since OHLCV not available
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,  # Needs computation
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[],
                )
            },
            existing_features_df=None,
        )
        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.return_value = coverage_analysis

        # Configure coverage evaluator to return features needing computation
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_computation.return_value = ['test_feature']
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_warmup.return_value = []

        # Configure feature store to raise an exception
        mock_dependencies['feature_store_port'].store_computed_features_offline.side_effect = RuntimeError("Database connection failed")

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None, "Fire-and-forget should return None even on failure"

        # Verify error notification was published
        mock_dependencies['message_publisher'].publish_preprocessing_error.assert_called_once()

        # Verify error details contain storage info
        call_args = mock_dependencies['message_publisher'].publish_preprocessing_error.call_args
        assert call_args.kwargs['request'] == request
        assert 'error_message' in call_args.kwargs
        error_message = call_args.kwargs['error_message']
        assert 'database' in error_message.lower() or 'failed' in error_message.lower()

    def test_empty_resampling_result_handles_gracefully(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        resampling_response_factory,
    ) -> None:
        """Test that empty resampling results are handled gracefully."""
        # Given
        request = request_builder.for_training().with_skip_existing(False).build()

        # Configure resampler to return empty data
        resampling_response = resampling_response_factory(candles_per_timeframe=0)
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify completion notification was published with zero features
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['total_features_computed'] == 0


class TestPreprocessingOrchestratorContextBehavior:
    """Test context-specific behavior (training, inference, backfill)."""

    def test_inference_context_forces_recompute(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test that inference context forces recomputation."""
        # Given
        request = request_builder.for_inference().build()

        # Configure successful processing
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify request has force_recompute enabled (inference default)
        assert request.force_recompute is True

        # Verify completion notification published
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['processing_context'] == "inference"


class TestPreprocessingOrchestratorSkipExisting:
    """Test skip_existing_features logic and early exit behavior."""

    def test_all_features_fully_covered_skips_computation(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
    ) -> None:
        """Test that fully covered features trigger early exit with skipped notification."""
        # Given
        import pandas as pd
        from datetime import datetime, timezone
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )

        request = request_builder.for_training().with_skip_existing(True).build()

        # Create coverage analysis indicating all features fully covered
        existing_features_df = pd.DataFrame({
            'test_feature': [1.0, 2.0, 3.0],
        })
        existing_features_df.index = pd.date_range('2023-01-01', periods=3, freq='5min', tz=timezone.utc)
        existing_features_df.index.name = 'event_timestamp'

        full_coverage = FeatureCoverageAnalysis(
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
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=True,
                    earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
                    record_count=288,
                    coverage_percentage=100.0,
                    missing_periods=[],
                )
            },
            existing_features_df=existing_features_df,
        )

        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.return_value = full_coverage
        mock_dependencies['feature_validator'].validate_definitions.return_value = {'test_feature': True}

        # Override the default mock to return no features needing computation (all are covered)
        mock_dependencies['feature_coverage_evaluator'].get_features_needing_computation.return_value = []

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None, "Fire-and-forget should return None"

        # Verify early exit - no resampling, computing, or storage occurred
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.assert_not_called()
        mock_dependencies['feature_computer'].compute_batch.assert_not_called()
        mock_dependencies['feature_store_port'].store_computed_features_offline.assert_not_called()

        # Verify completion notification with skipped=True
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()
        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['request'] == request
        assert call_args.kwargs['total_features_computed'] == 0
        assert call_args.kwargs['timeframes_processed'] == []
        assert call_args.kwargs['success_details']['skipped'] is True
        assert call_args.kwargs['success_details']['reason'] == "all_features_exist"

    def test_no_ohlcv_data_forces_resampling_and_computation(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test that missing OHLCV data forces resampling from base timeframe."""
        # Given
        from datetime import datetime, timezone
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )

        request = request_builder.for_training().with_skip_existing(True).build()

        # Coverage analysis: no OHLCV data available but features DON'T need warmup
        # (warmup only needed if missing_periods or coverage < 100%)
        no_ohlcv_coverage = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_5,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=False,  # Forces resampling
            ohlcv_earliest_timestamp=None,
            ohlcv_latest_timestamp=None,
            ohlcv_record_count=0,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Since OHLCV not available, must resample
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=True,  # Fully covered (no warmup needed)
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=100.0,  # 100% coverage (no warmup needed)
                    missing_periods=[],  # No missing periods (no warmup needed)
                ),
            },
            existing_features_df=None,
        )

        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.return_value = no_ohlcv_coverage
        mock_dependencies['feature_validator'].validate_definitions.return_value = {'test_feature': True}

        # Configure resampler and computer
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response
        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify resampling occurred (OHLCV not available, must resample from base)
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.assert_called_once()

        # Verify feature computation occurred
        mock_dependencies['feature_computer'].compute_batch.assert_called()

        # Verify completion notification published
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

    def test_warmup_failure_publishes_error_notification(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        resampling_response_factory,
    ) -> None:
        """Test that warmup failure triggers error notification and early exit."""
        # Given
        from unittest.mock import patch

        request = request_builder.for_training().with_skip_existing(False).build()

        mock_dependencies['feature_validator'].validate_definitions.return_value = {'test_feature': True}

        # Configure resampler to succeed
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        # Mock _handle_feature_warmup to return False (warmup failure)
        with patch.object(preprocessing_orchestrator, '_handle_feature_warmup', return_value=False):
            # When
            result = preprocessing_orchestrator.process_feature_computation_request(request)

            # Then
            assert result is None, "Fire-and-forget should return None even on failure"

            # Verify early exit - no feature computation or storage occurred after warmup failure
            mock_dependencies['feature_computer'].compute_batch.assert_not_called()
            mock_dependencies['feature_store_port'].store_computed_features_offline.assert_not_called()

            # Verify error notification was published with warmup failure details
            mock_dependencies['message_publisher'].publish_preprocessing_error.assert_called_once()
            call_args = mock_dependencies['message_publisher'].publish_preprocessing_error.call_args
            assert call_args.kwargs['request'] == request
            assert call_args.kwargs['error_message'] == "Feature warmup failed"
            assert call_args.kwargs['error_details']['failed_step'] == "feature_warmup"
            assert call_args.kwargs['failed_step'] == "feature_warmup"

            # Verify NO completion notification was published
            mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_not_called()


class TestPreprocessingOrchestratorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_partial_timeframe_failure_still_publishes_successful_timeframes(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test that partial failures still publish results for successful timeframes."""
        # Given
        request = (request_builder
                   .for_training()
                   .with_skip_existing(False)
                   .with_target_timeframes([Timeframe.MINUTE_5, Timeframe.MINUTE_15])
                   .build())

        # Mock coverage analysis to require resampling for both timeframes
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_5m = FeatureCoverageAnalysis(
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
            requires_resampling=True,  # Force resampling
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
            },
            existing_features_df=None,
        )

        coverage_15m = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_15,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=96,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Force resampling
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
            },
            existing_features_df=None,
        )

        # Mock analyzer to return different coverage for different timeframes
        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.side_effect = [coverage_5m, coverage_15m]

        # Configure resampler to return data for both timeframes
        resampling_response = resampling_response_factory(
            timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15]
        )
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        # Configure feature computer to succeed for first call, fail for second
        mock_dependencies['feature_computer'].compute_batch.side_effect = [
            sample_features_df,  # First timeframe succeeds
            RuntimeError("Computation failed for second timeframe"),  # Second fails
        ]

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify error notification was published (partial failure)
        mock_dependencies['message_publisher'].publish_preprocessing_error.assert_called_once()

        call_args = mock_dependencies['message_publisher'].publish_preprocessing_error.call_args
        assert call_args.kwargs['request'] == request
        error_message = call_args.kwargs['error_message']
        assert 'failed' in error_message.lower() or 'error' in error_message.lower()

    def test_request_with_large_feature_set_handles_correctly(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        resampling_response_factory,
    ) -> None:
        """Test processing large feature sets (stress test)."""
        # Given
        from drl_trading_common.config.feature_config import FeatureDefinition

        # Create 100 features
        large_feature_set = [
            FeatureDefinition(
                name=f"feature_{i}",
                enabled=True,
                derivatives=[0],
                parameter_sets=[]
            )
            for i in range(100)
        ]

        request = (request_builder
                   .for_training()
                   .with_skip_existing(False)
                   .with_features(large_feature_set)
                   .build())

        # Mock coverage analysis to require resampling
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_analysis = FeatureCoverageAnalysis(
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
            requires_resampling=True,  # Force resampling
            feature_coverage={
                f"feature_{i}": FeatureCoverageInfo(
                    feature_name=f"feature_{i}",
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
                for i in range(100)
            },
            existing_features_df=None,
        )

        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.return_value = coverage_analysis

        # Configure successful processing
        resampling_response = resampling_response_factory()
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        # All features valid
        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            f"feature_{i}": True for i in range(100)
        }

        # Create large feature DataFrame (100 features)
        import pandas as pd
        large_df = pd.DataFrame({
            f"feature_{i}": [1.0, 2.0, 3.0] for i in range(100)
        })
        large_df.index = pd.date_range('2023-01-01', periods=3, freq='5min')
        large_df.index.name = 'timestamp'
        mock_dependencies['feature_computer'].compute_batch.return_value = large_df

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify completion with correct feature count
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        assert call_args.kwargs['total_features_computed'] == 100

    def test_concurrent_timeframe_processing_with_parallel_enabled(
        self,
        preprocessing_orchestrator: "PreprocessingOrchestrator",
        mock_dependencies: dict[str, "Mock"],
        request_builder: "FeaturePreprocessingRequestBuilder",
        sample_features_df: "DataFrame",
        resampling_response_factory,
    ) -> None:
        """Test parallel processing of multiple timeframes."""
        # Given
        request = (request_builder
                   .for_training()
                   .with_skip_existing(False)
                   .with_parallel_processing(True)
                   .with_target_timeframes([Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.MINUTE_30])
                   .build())

        # Mock coverage analysis to require resampling for all timeframes
        from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import (
            FeatureCoverageAnalysis,
            FeatureCoverageInfo,
        )
        from datetime import datetime, timezone

        coverage_5m = FeatureCoverageAnalysis(
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
            requires_resampling=True,  # Force resampling
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
            },
            existing_features_df=None,
        )

        coverage_15m = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_15,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=96,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Force resampling
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
            },
            existing_features_df=None,
        )

        coverage_30m = FeatureCoverageAnalysis(
            symbol="BTCUSD",
            timeframe=Timeframe.MINUTE_30,
            requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_available=True,
            ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
            ohlcv_record_count=48,
            adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
            requires_resampling=True,  # Force resampling
            feature_coverage={
                'test_feature': FeatureCoverageInfo(
                    feature_name='test_feature',
                    is_fully_covered=False,
                    earliest_timestamp=None,
                    latest_timestamp=None,
                    record_count=0,
                    coverage_percentage=0.0,
                    missing_periods=[(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 2, tzinfo=timezone.utc))],
                )
            },
            existing_features_df=None,
        )

        # Mock analyzer to return coverage for each timeframe
        mock_dependencies['feature_coverage_analyzer'].analyze_feature_coverage.side_effect = [coverage_5m, coverage_15m, coverage_30m]

        # Configure resampler to return data for all timeframes
        resampling_response = resampling_response_factory(
            timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.MINUTE_30],
            candles_per_timeframe=5
        )
        mock_dependencies['market_data_resampler'].resample_symbol_data_incremental.return_value = resampling_response

        mock_dependencies['feature_validator'].validate_definitions.return_value = {
            'test_feature': True
        }

        mock_dependencies['feature_computer'].compute_batch.return_value = sample_features_df

        # When
        result = preprocessing_orchestrator.process_feature_computation_request(request)

        # Then
        assert result is None

        # Verify completion with all timeframes
        mock_dependencies['message_publisher'].publish_preprocessing_completed.assert_called_once()

        call_args = mock_dependencies['message_publisher'].publish_preprocessing_completed.call_args
        timeframes_processed = call_args.kwargs['timeframes_processed']
        assert len(timeframes_processed) == 3
        assert Timeframe.MINUTE_5 in timeframes_processed
        assert Timeframe.MINUTE_15 in timeframes_processed
        assert Timeframe.MINUTE_30 in timeframes_processed

        # Total features = 2 features * 3 timeframes = 6
        assert call_args.kwargs['total_features_computed'] == 6
