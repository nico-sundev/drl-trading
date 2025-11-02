"""
Shared fixtures for PreprocessService unit tests.

This conftest provides reusable fixtures for testing the PreprocessService,
including mock dependencies and test data builders.
"""
from typing import Callable
from datetime import datetime, timezone

import pytest
from unittest.mock import Mock
from pandas import DataFrame

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator
from drl_trading_preprocess.core.service.compute.computing_service import FeatureComputingService
from drl_trading_preprocess.core.service.validate.feature_validator import FeatureValidator
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService
from drl_trading_preprocess.core.service.coverage.feature_coverage_analyzer import FeatureCoverageAnalyzer
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort
from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import PreprocessingMessagePublisherPort
from drl_trading_preprocess.core.model.resample.resampling_response import ResamplingResponse
from drl_trading_core.common.model.market_data_model import MarketDataModel
from builders import FeaturePreprocessingRequestBuilder

@pytest.fixture
def request_builder() -> FeaturePreprocessingRequestBuilder:
    """Provides a builder for creating test requests."""
    return FeaturePreprocessingRequestBuilder()


@pytest.fixture
def mock_market_data_resampler() -> Mock:
    """Mock MarketDataResamplingService."""
    mock = Mock(spec=MarketDataResamplingService)
    # Default behavior: return empty response
    mock.resample_symbol_data_incremental.return_value = Mock(resampled_data={})
    return mock


@pytest.fixture
def mock_feature_computer() -> Mock:
    """Mock FeatureComputingService."""
    mock = Mock(spec=FeatureComputingService)
    # Default behavior: return empty DataFrame
    mock.compute_batch.return_value = DataFrame()
    return mock


@pytest.fixture
def mock_feature_validator() -> Mock:
    """Mock FeatureValidator."""
    mock = Mock(spec=FeatureValidator)
    # Default behavior: all features valid
    mock.validate_definitions.return_value = {}
    return mock


@pytest.fixture
def mock_feature_store_port() -> Mock:
    """Mock IFeatureStoreSavePort."""
    mock = Mock(spec=IFeatureStoreSavePort)
    # Default behavior: successful storage
    mock.store_computed_features_offline.return_value = None
    mock.batch_materialize_features.return_value = None
    return mock


@pytest.fixture
def mock_feature_coverage_analyzer() -> Mock:
    """Mock FeatureCoverageAnalyzer."""
    mock = Mock(spec=FeatureCoverageAnalyzer)
    # Default behavior: no existing features (compute all)
    from drl_trading_preprocess.core.model.coverage.feature_coverage_analysis import FeatureCoverageAnalysis
    from datetime import datetime, timezone

    default_analysis = FeatureCoverageAnalysis(
        symbol="BTCUSD",
        timeframe=Timeframe.MINUTE_5,
        requested_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        requested_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
        ohlcv_available=True,
        ohlcv_earliest_timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
        ohlcv_latest_timestamp=datetime(2023, 1, 2, tzinfo=timezone.utc),
        ohlcv_record_count=288,  # 24 hours * 12 (5-min candles per hour)
        adjusted_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        adjusted_end_time=datetime(2023, 1, 2, tzinfo=timezone.utc),
        feature_coverage={},
        existing_features_df=None,
    )
    mock.analyze_feature_coverage.return_value = default_analysis
    return mock


@pytest.fixture
def mock_message_publisher() -> Mock:
    """
    Mock PreprocessingMessagePublisherPort.

    This is the PRIMARY mock we care about in these tests,
    since we're testing the fire-and-forget notification behavior.
    """
    mock = Mock(spec=PreprocessingMessagePublisherPort)
    mock.publish_preprocessing_completed.return_value = None
    mock.publish_preprocessing_error.return_value = None
    mock.publish_feature_validation_error.return_value = None
    mock.health_check.return_value = True
    return mock


@pytest.fixture
def mock_dependencies(
    mock_market_data_resampler: Mock,
    mock_feature_computer: Mock,
    mock_feature_validator: Mock,
    mock_feature_store_port: Mock,
    mock_feature_coverage_analyzer: Mock,
    mock_message_publisher: Mock,
) -> dict:
    """
    Provides all mock dependencies as a dictionary.

    This is useful when you need to access individual mocks
    to verify their behavior or set up specific test scenarios.
    """
    return {
        'market_data_resampler': mock_market_data_resampler,
        'feature_computer': mock_feature_computer,
        'feature_validator': mock_feature_validator,
        'feature_store_port': mock_feature_store_port,
        'feature_coverage_analyzer': mock_feature_coverage_analyzer,
        'message_publisher': mock_message_publisher,
    }


@pytest.fixture
def preprocessing_orchestrator(mock_dependencies: dict) -> PreprocessingOrchestrator:
    """
    Create PreprocessingOrchestrator with all mocked dependencies.

    The orchestrator uses REAL internal logic, but all external calls
    go through mocks. This allows us to test the orchestration
    logic without hitting real databases or message queues.
    """
    from drl_trading_preprocess.infrastructure.config.preprocess_config import DaskConfigs
    from drl_trading_common.config.dask_config import DaskConfig

    # Create default DaskConfigs for tests (synchronous scheduler for deterministic behavior)
    dask_configs = DaskConfigs(
        coverage_analysis=DaskConfig(
            scheduler="synchronous",
            num_workers=1,
            threads_per_worker=1,
            memory_limit_per_worker_mb=512,
        ),
        feature_computation=DaskConfig(
            scheduler="synchronous",
            num_workers=1,
            threads_per_worker=1,
            memory_limit_per_worker_mb=512,
        ),
    )

    return PreprocessingOrchestrator(
        market_data_resampler=mock_dependencies['market_data_resampler'],
        feature_computer=mock_dependencies['feature_computer'],
        feature_validator=mock_dependencies['feature_validator'],
        feature_store_port=mock_dependencies['feature_store_port'],
        feature_coverage_analyzer=mock_dependencies['feature_coverage_analyzer'],
        message_publisher=mock_dependencies['message_publisher'],
        dask_configs=dask_configs,
    )


@pytest.fixture
def sample_features_df() -> DataFrame:
    """
    Create a sample DataFrame representing computed features.

    This is useful for setting up mock return values from
    feature computation or retrieval operations.
    """
    import pandas as pd

    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0],
        'feature_2': [4.0, 5.0, 6.0],
    })
    # Set timestamp index
    df.index = pd.date_range('2023-01-01', periods=3, freq='5min')
    df.index.name = 'timestamp'

    return df


@pytest.fixture
def market_data_factory() -> Callable:
    """
    Factory for creating realistic MarketDataModel instances.

    Eliminates duplication of MarketDataModel construction across tests
    while providing realistic price data based on actual BTC trading patterns.

    Example:
        market_data = market_data_factory(symbol="ETHUSD", timeframe=Timeframe.MINUTE_15)
        market_data_custom = market_data_factory(open_price=43500.0, volume=3000)
    """
    def _create(
        symbol: str = "BTCUSD",
        timeframe: Timeframe = Timeframe.MINUTE_5,
        timestamp: datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        **overrides
    ) -> MarketDataModel:
        """
        Create a MarketDataModel with realistic defaults.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            timestamp: Candle timestamp
            **overrides: Override any field (open_price, high_price, low_price, close_price, volume)

        Returns:
            MarketDataModel with realistic BTC price data
        """
        defaults = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": timestamp,
            # Realistic BTC price data (±0.5% intracandle movement)
            "open_price": 43127.85,
            "high_price": 43156.22,
            "low_price": 43089.44,
            "close_price": 43143.91,
            "volume": 2847,  # Realistic 5m BTC volume
        }
        defaults.update(overrides)
        return MarketDataModel(**defaults)

    return _create


@pytest.fixture
def resampling_response_factory(market_data_factory: Callable) -> Callable:
    """
    Factory for creating ResamplingResponse instances.

    Simplifies test setup by providing a clean API for creating
    resampling responses with multiple timeframes.

    Example:
        response = resampling_response_factory(
            timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15],
            candles_per_timeframe=2
        )
    """
    def _create(
        symbol: str = "BTCUSD",
        base_timeframe: Timeframe = Timeframe.MINUTE_1,
        timeframes: list[Timeframe] | None = None,
        candles_per_timeframe: int = 1,
        **overrides
    ) -> ResamplingResponse:
        """
        Create a ResamplingResponse with realistic data.

        Args:
            symbol: Trading pair symbol
            base_timeframe: Source timeframe for resampling
            timeframes: Target timeframes (defaults to [MINUTE_5])
            candles_per_timeframe: Number of candles to create per timeframe
            **overrides: Override any ResamplingResponse field

        Returns:
            ResamplingResponse with market data for all timeframes
        """
        if timeframes is None:
            timeframes = [Timeframe.MINUTE_5]

        resampled_data = {}
        new_candles_count = {}

        for tf in timeframes:
            candles = []
            for i in range(candles_per_timeframe):
                timestamp = datetime(2023, 1, 1, 0, i * 5, 0, tzinfo=timezone.utc)
                candles.append(market_data_factory(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=timestamp,
                ))
            resampled_data[tf] = candles
            new_candles_count[tf] = candles_per_timeframe

        defaults = {
            "symbol": symbol,
            "base_timeframe": base_timeframe,
            "resampled_data": resampled_data,
            "new_candles_count": new_candles_count,
            "processing_start_time": datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "processing_end_time": datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "source_records_processed": candles_per_timeframe * len(timeframes),
        }
        defaults.update(overrides)
        return ResamplingResponse(**defaults)

    return _create
