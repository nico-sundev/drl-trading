"""Unit tests for the stateful resampling architecture."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService
from drl_trading_preprocess.adapter.resampling.noop_state_persistence_service import NoOpStatePersistenceService
from drl_trading_preprocess.infrastructure.config.preprocess_config import ResampleConfig


@pytest.fixture
def resample_config() -> ResampleConfig:
    """Create a complete ResampleConfig for testing."""
    return ResampleConfig(
        historical_start_date=datetime(2020, 1, 1),
        max_batch_size=1000,
        progress_log_interval=1,
        enable_incomplete_candle_publishing=True,
        chunk_size=100,
        memory_warning_threshold_mb=500,
        pagination_limit=1000,
        max_memory_usage_mb=1000,
        state_persistence_enabled=False,
        state_file_path="test_state.json",
        state_backup_interval=3600,
        auto_cleanup_inactive_symbols=True,
        inactive_symbol_threshold_hours=24
    )


class TestStatefulResamplingArchitecture:
    """Test the complete stateful resampling architecture."""

    @pytest.fixture
    def sample_market_data(self) -> list[MarketDataModel]:
        """Create sample market data for testing."""
        # Given
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        return [
            MarketDataModel(
                symbol="BTCUSDT",
                timeframe=Timeframe.MINUTE_1,
                timestamp=base_time + timedelta(minutes=i),
                open_price=100.0 + i,
                high_price=101.0 + i,
                low_price=99.0 + i,
                close_price=100.5 + i,
                volume=1000 + i * 10
            ) for i in range(10)
        ]

    @pytest.fixture
    def mock_dependencies(self, resample_config: ResampleConfig) -> dict:
        """Create mock dependencies for the resampling service."""
        # Given
        market_data_reader = Mock()
        message_publisher = Mock()
        candle_accumulator_service = Mock()
        state_persistence = NoOpStatePersistenceService()

        return {
            'market_data_reader': market_data_reader,
            'message_publisher': message_publisher,
            'candle_accumulator_service': candle_accumulator_service,
            'resample_config': resample_config,
            'state_persistence': state_persistence,
        }

    def test_resampling_service_can_be_instantiated(self, mock_dependencies: dict) -> None:
        """Test that MarketDataResamplingService can be created with current dependencies."""
        # Given
        # Dependencies from fixture

        # When
        service = MarketDataResamplingService(**mock_dependencies)

        # Then
        assert service is not None
        assert service.market_data_reader is not None
        assert service.message_publisher is not None
        assert service.candle_accumulator_service is not None
        assert service.resample_config is not None

    def test_configuration_can_be_extended_for_stateful_processing(self, resample_config: ResampleConfig) -> None:
        """Test that configuration can be extended for future stateful processing."""
        # Given
        config = resample_config

        # When & Then
        # Current config should work
        assert config is not None

        # These will be added in future implementation
        # assert hasattr(config, 'pagination_limit')  # To be added
        # assert hasattr(config, 'state_persistence_enabled')  # To be added

    def test_memory_efficient_pagination_processing(self, mock_dependencies: dict, sample_market_data: list[MarketDataModel]) -> None:
        """Test that pagination configuration is properly set up."""
        # Given
        # Add a mock pagination_limit to config for this test
        mock_dependencies['resample_config'].pagination_limit = 3  # Small limit for testing

        # When
        service = MarketDataResamplingService(**mock_dependencies)

        # Then
        # Verify pagination configuration is accessible
        assert hasattr(service.resample_config, 'pagination_limit')
        assert service.resample_config.pagination_limit == 3

        # Verify the service is ready for pagination (has required attributes)
        assert hasattr(service, 'resample_config')
        assert hasattr(service, 'market_data_reader')

        # This test validates the foundation is ready for pagination implementation
        # The actual pagination methods will be implemented in the next phase

    def test_current_resampling_still_works(self, mock_dependencies: dict, sample_market_data: list[MarketDataModel]) -> None:
        """Test that current resampling functionality still works while we prepare for stateful changes."""
        # Given
        mock_dependencies['market_data_reader'].get_symbol_data_range.return_value = sample_market_data
        mock_dependencies['market_data_reader'].get_symbol_data_range_paginated.return_value = sample_market_data
        service = MarketDataResamplingService(**mock_dependencies)

        # When
        response = service.resample_symbol_data_incremental(
            symbol="BTCUSDT",
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5]
        )

        # Then
        assert response is not None
        assert response.symbol == "BTCUSDT"
        # The response should be properly formed even if we haven't implemented full stateful processing yet
