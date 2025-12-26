"""
Unit tests for context-based stateful vs stateless resampling behavior.

This test validates that the resampling service correctly distinguishes between:
- Backfill mode: Stateless (uses configured time range, ignores state)
- Inference/Training mode: Stateful (uses incremental processing with state)
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from drl_trading_common.core.model.processing_context import ProcessingContext
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import (
    MarketDataResamplingService,
)
from drl_trading_preprocess.core.service.resample.candle_accumulator_service import (
    CandleAccumulatorService,
)
from drl_trading_preprocess.application.config.preprocess_config import (
    ResampleConfig,
)


class TestContextBasedResampling:
    """Test context-based stateful vs stateless behavior."""

    @pytest.fixture
    def mock_market_data_fetch_port(self) -> Mock:
        """Mock market data fetch port."""
        mock = Mock()
        # Service uses get_symbol_data_range_paginated, not fetch_market_data_paginated
        mock.get_symbol_data_range_paginated = Mock(return_value=[])
        return mock

    @pytest.fixture
    def mock_message_publisher(self) -> Mock:
        """Mock message publisher port."""
        mock = Mock()
        mock.publish_resampled_data = Mock()
        mock.publish_resampling_error = Mock()
        return mock

    @pytest.fixture
    def mock_state_persistence(self) -> Mock:
        """Mock state persistence service."""
        mock = Mock()
        mock.save_context = Mock(return_value=True)
        mock.auto_save_if_needed = Mock(return_value=True)  # Used by _save_context_if_enabled
        mock.load_context = Mock(return_value=None)
        return mock

    @pytest.fixture
    def candle_accumulator_service(self) -> CandleAccumulatorService:
        """Create candle accumulator service."""
        return CandleAccumulatorService()

    @pytest.fixture
    def market_data_resampling_service(
        self,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
        mock_state_persistence: Mock,
        candle_accumulator_service: CandleAccumulatorService,
        resample_config: ResampleConfig,
    ) -> MarketDataResamplingService:
        """Create resampling service with mocked dependencies."""
        return MarketDataResamplingService(
            market_data_reader=mock_market_data_fetch_port,
            message_publisher=mock_message_publisher,
            candle_accumulator_service=candle_accumulator_service,
            resample_config=resample_config,
            state_persistence=mock_state_persistence,
        )

    def test_backfill_mode_ignores_state_uses_configured_time_range(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test backfill mode uses configured time range and ignores last_processed_timestamp.

        Given:
        - Service has state with last_processed_timestamp = 2024-01-15
        - Processing context = "backfill"
        - Configured historical_start_date = 2024-01-01

        When:
        - resample_symbol_data_incremental is called with processing_context="backfill"

        Then:
        - Should fetch data from configured start date (2024-01-01), not from state (2024-01-15)
        - Demonstrates stateless behavior for reproducible backfills
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Simulate existing state (service has already processed data up to 2024-01-15)
        last_processed = datetime(2024, 1, 15, tzinfo=timezone.utc)
        market_data_resampling_service.context.update_last_processed_timestamp(
            symbol, base_timeframe, last_processed
        )

        # Mock data fetch to return empty (we only care about the start_time parameter)
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.BACKFILL.value  # Backfill mode
        )

        # Then
        # Verify fetch was called with configured start date, not last_processed_timestamp
        fetch_call = mock_market_data_fetch_port.get_symbol_data_range_paginated.call_args
        actual_start_time = fetch_call.kwargs['start_time']

        # Should use configured historical_start_date (not the state's last_processed_timestamp)
        expected_start_time = market_data_resampling_service.resample_config.historical_start_date
        assert actual_start_time == expected_start_time, (
            f"Backfill mode should use configured start date {expected_start_time}, "
            f"not state's last_processed_timestamp {last_processed}"
        )

    def test_inference_mode_uses_state_for_incremental_processing(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test inference mode uses state for incremental processing.

        Given:
        - Service has state with last_processed_timestamp = 2024-01-15
        - Processing context = "inference"
        - Configured historical_start_date = 2024-01-01

        When:
        - resample_symbol_data_incremental is called with processing_context="inference"

        Then:
        - Should fetch data from last_processed_timestamp (2024-01-15), not configured start
        - Demonstrates stateful behavior for incremental updates
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Simulate existing state
        last_processed = datetime(2024, 1, 15, tzinfo=timezone.utc)
        market_data_resampling_service.context.update_last_processed_timestamp(
            symbol, base_timeframe, last_processed
        )

        # Mock data fetch to return empty
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.INFERENCE.value  # Inference mode
        )

        # Then
        # Verify fetch was called with last_processed_timestamp
        fetch_call = mock_market_data_fetch_port.get_symbol_data_range_paginated.call_args
        actual_start_time = fetch_call.kwargs['start_time']

        assert actual_start_time == last_processed, (
            f"Inference mode should use last_processed_timestamp {last_processed}, "
            f"not configured start date"
        )

    def test_training_mode_uses_state_for_incremental_processing(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test training mode uses state for incremental processing.

        Given:
        - Service has state with last_processed_timestamp = 2024-01-10
        - Processing context = "training"

        When:
        - resample_symbol_data_incremental is called with processing_context="training"

        Then:
        - Should fetch data from last_processed_timestamp (2024-01-10)
        - Demonstrates stateful behavior for training data updates
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Simulate existing state
        last_processed = datetime(2024, 1, 10, tzinfo=timezone.utc)
        market_data_resampling_service.context.update_last_processed_timestamp(
            symbol, base_timeframe, last_processed
        )

        # Mock data fetch to return empty
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.TRAINING.value  # Training mode
        )

        # Then
        # Verify fetch was called with last_processed_timestamp
        fetch_call = mock_market_data_fetch_port.get_symbol_data_range_paginated.call_args
        actual_start_time = fetch_call.kwargs['start_time']

        assert actual_start_time == last_processed, (
            f"Training mode should use last_processed_timestamp {last_processed}"
        )

    def test_default_context_uses_stateful_behavior(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test default context (when not specified) uses stateful behavior.

        Given:
        - Service has state with last_processed_timestamp
        - No processing_context specified (uses default "inference")

        When:
        - resample_symbol_data_incremental is called without processing_context

        Then:
        - Should default to stateful behavior (use last_processed_timestamp)
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Simulate existing state
        last_processed = datetime(2024, 1, 20, tzinfo=timezone.utc)
        market_data_resampling_service.context.update_last_processed_timestamp(
            symbol, base_timeframe, last_processed
        )

        # Mock data fetch to return empty
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # When - Call without processing_context parameter
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            # processing_context not specified, uses default "inference"
        )

        # Then
        # Verify fetch was called with last_processed_timestamp (stateful behavior)
        fetch_call = mock_market_data_fetch_port.get_symbol_data_range_paginated.call_args
        actual_start_time = fetch_call.kwargs['start_time']

        assert actual_start_time == last_processed, (
            "Default context should use stateful behavior (last_processed_timestamp)"
        )

    def test_backfill_mode_resets_accumulators_for_reproducibility(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test backfill mode resets accumulators to ensure reproducible results.

        Given:
        - Service has active accumulators with state from previous processing
        - Processing context = "backfill"

        When:
        - resample_symbol_data_incremental is called with processing_context="backfill"

        Then:
        - Accumulators should be reset before processing
        - Ensures reproducible results regardless of previous service state
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Simulate existing accumulator state (service has processed data before)
        # Manually add accumulator to context to simulate previous processing
        accumulator = market_data_resampling_service.context.get_accumulator(
            symbol, Timeframe.MINUTE_5
        )
        # Simulate partial candle state
        accumulator.current_period_start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        accumulator.open_price = 100.0
        accumulator.high_price = 101.0
        accumulator.low_price = 99.0
        accumulator.close_price = 100.5
        accumulator.volume = 1000
        accumulator.record_count = 3

        # Verify accumulator has state before backfill
        assert accumulator.current_period_start is not None
        assert accumulator.record_count == 3

        # Mock data fetch to return empty
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.BACKFILL.value  # Backfill mode
        )

        # Then
        # Verify accumulator was reset (should not exist in active accumulators anymore)
        # Getting it again should create a fresh one
        new_accumulator = market_data_resampling_service.context.get_accumulator(
            symbol, Timeframe.MINUTE_5
        )

        # New accumulator should be fresh (no state)
        assert new_accumulator.current_period_start is None, (
            "Backfill mode should reset accumulators to None"
        )
        assert new_accumulator.record_count == 0, (
            "Backfill mode should reset accumulator record count to 0"
        )

    def test_backfill_mode_does_not_save_context_state(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test backfill mode does NOT save context state for stateless operation.

        Given:
        - State persistence is enabled
        - Processing context = "backfill"

        When:
        - resample_symbol_data_incremental is called with processing_context="backfill"

        Then:
        - State persistence save_context should NOT be called
        - Ensures backfill mode remains truly stateless
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Mock data fetch to return empty
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = []

        # Spy on state persistence service
        mock_state_persistence = market_data_resampling_service.state_persistence

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.BACKFILL.value  # Backfill mode
        )

        # Then
        # Verify save_context was NOT called (backfill should be stateless)
        mock_state_persistence.auto_save_if_needed.assert_not_called()

    def test_inference_mode_saves_context_state(
        self,
        market_data_resampling_service: MarketDataResamplingService,
        mock_market_data_fetch_port: Mock,
        mock_message_publisher: Mock,
    ) -> None:
        """
        Test inference mode DOES save context state for stateful operation.

        Given:
        - State persistence is enabled
        - Processing context = "inference"

        When:
        - resample_symbol_data_incremental is called with processing_context="inference"

        Then:
        - State persistence save_context SHOULD be called
        - Ensures inference mode maintains state across requests
        """
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # Mock data fetch to return some data (so save happens)
        from drl_trading_core.core.model.market_data_model import MarketDataModel
        mock_data = [
            MarketDataModel(
                symbol=symbol,
                timeframe=base_timeframe,
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open_price=100.0,
                high_price=101.0,
                low_price=99.0,
                close_price=100.5,
                volume=1000
            )
        ]
        mock_market_data_fetch_port.get_symbol_data_range_paginated.return_value = mock_data

        # Spy on state persistence service
        mock_state_persistence = market_data_resampling_service.state_persistence

        # When
        market_data_resampling_service.resample_symbol_data_incremental(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            processing_context=ProcessingContext.INFERENCE.value  # Inference mode
        )

        # Then
        # Verify save_context WAS called (inference should be stateful)
        mock_state_persistence.auto_save_if_needed.assert_called_once()
