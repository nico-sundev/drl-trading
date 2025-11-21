from datetime import datetime

import pytest
from injector import Injector

from drl_trading_adapter.infrastructure.di.adapter_module import AdapterModule
from drl_trading_common.config.feature_config import (
    FeatureStoreConfig,
)
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import PreprocessConfig
from drl_trading_preprocess.infrastructure.di.preprocess_module import PreprocessModule


@pytest.fixture(scope="function")
def real_feast_container(
    feature_store_config: FeatureStoreConfig,
    temp_feast_repo: str,
) -> Injector:
    """Create a dependency injection container with REAL Feast integration.

    This fixture provides a configured injector instance with real services
    for true integration testing. Uses minimal test features that don't depend
    on the strategy module, maintaining proper dependency boundaries.

    Args:
        test_feature_factory: Test feature factory with RSI and close price features
        config_fixture: Test configuration with temp Feast repository
        temp_feast_repo: Path to the temporary Feast repository

    Returns:
        Injector: Configured DI container with real services and test features
    """

    # Create injector with the real Feast configuration
    app_module = AdapterModule()
    from drl_trading_preprocess.infrastructure.config.preprocess_config import FeatureComputationConfig, ResampleConfig
    preprocess_module = PreprocessModule(PreprocessConfig(
        feature_store_config=feature_store_config,
        feature_computation_config=FeatureComputationConfig(warmup_candles=10),
        resample_config=ResampleConfig(
            state_persistence_enabled=False,
            historical_start_date=datetime(2020, 1, 1),
            max_batch_size=1000,
            progress_log_interval=10,
            enable_incomplete_candle_publishing=False,
            chunk_size=100,
            memory_warning_threshold_mb=100,
            pagination_limit=1000,
            max_memory_usage_mb=500,
            state_file_path="/tmp/test_state.json",
            state_backup_interval=60,
            auto_cleanup_inactive_symbols=False,
            inactive_symbol_threshold_hours=24,
        )
    ))
    injector = Injector([app_module, preprocess_module])

    return injector


@pytest.fixture
def feature_version_info_fixture() -> FeatureConfigVersionInfo:
    """Create feature version info for integration testing."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="integration_test_hash_123",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[
            {"name": "rsi_14", "enabled": True, "role": "observation_space"},
            {"name": "close_1", "enabled": True, "role": "observation_space"},
            {"name": "reward", "enabled": True, "role": "reward_engineering"},
            {
                "name": "cumulative_return",
                "enabled": True,
                "role": "reward_engineering",
            },
        ],
    )
