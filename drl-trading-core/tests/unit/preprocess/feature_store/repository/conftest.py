"""
Test fixtures for feature_store_save_repository tests.

Provides common test fixtures and mocks for testing the FeatureStoreSaveRepository class.
"""

import tempfile
from datetime import datetime
from typing import Generator
from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_common.model.timeframe import Timeframe
from feast import FeatureService, FeatureStore
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
    FeatureStoreSaveRepository,
)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def feature_store_config(temp_dir: str) -> FeatureStoreConfig:
    """Create a test feature store configuration."""
    return FeatureStoreConfig(
        enabled=True,
        repo_path=temp_dir,
        offline_store_path=f"{temp_dir}/offline_store.parquet",
        entity_name="trading_entity",
        ttl_days=30,
        online_enabled=True,
        service_name="test_service",
        service_version="1.0.0"
    )


@pytest.fixture
def sample_features_df() -> DataFrame:
    """Create sample features DataFrame for testing."""
    return DataFrame({
        "event_timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ],
        "rsi_14": [30.5, 45.2, 67.8],
        "sma_20": [1.0850, 1.0855, 1.0860],
        "bb_upper": [1.0870, 1.0875, 1.0880]
    })


@pytest.fixture
def eurusd_h1_dataset_id() -> DatasetIdentifier:
    """Create a sample dataset identifier for EUR/USD H1."""
    return DatasetIdentifier(
        symbol="EURUSD",
        timeframe=Timeframe.HOUR_1
    )


@pytest.fixture
def gbpusd_m15_dataset_id() -> DatasetIdentifier:
    """Create a sample dataset identifier for GBP/USD M15."""
    return DatasetIdentifier(
        symbol="GBPUSD",
        timeframe=Timeframe.MINUTE_15
    )


@pytest.fixture
def mock_feast_provider() -> Mock:
    """Create a mock FeastProvider for testing."""
    mock_provider = Mock()
    mock_provider.get_feature_store.return_value = Mock()
    mock_provider.create_feature_view.return_value = Mock()
    mock_provider.create_feature_service.return_value = Mock()
    return mock_provider


@pytest.fixture
def mock_feature_store() -> Mock:
    """Create a mock FeatureStore for testing."""
    mock_store = Mock(spec=FeatureStore)

    # Mock online features response
    mock_online_response = Mock()
    mock_online_response.to_df.return_value = DataFrame({
        "symbol": ["EURUSD"],
        "timeframe": ["H1"],
        "rsi_14": [45.2],
        "sma_20": [1.0855]
    })
    mock_store.get_online_features.return_value = mock_online_response

    # Mock historical features response
    mock_historical_response = Mock()
    mock_historical_response.to_df.return_value = DataFrame({
        "event_timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ],
        "symbol": ["EURUSD", "EURUSD", "EURUSD"],
        "timeframe": ["H1", "H1", "H1"],
        "rsi_14": [30.5, 45.2, 67.8],
        "sma_20": [1.0850, 1.0855, 1.0860]
    })
    mock_store.get_historical_features.return_value = mock_historical_response

    return mock_store


@pytest.fixture
def mock_feature_service() -> Mock:
    """Create a mock FeatureService for testing."""
    return Mock(spec=FeatureService)


@pytest.fixture
def mock_offline_repo() -> Mock:
    """Create a mock OfflineFeatureRepoInterface for testing."""
    mock_repo = Mock()
    mock_repo.store_features_incrementally.return_value = 3  # Default to storing 3 features
    mock_repo.feature_exists.return_value = False
    mock_repo.get_feature_count.return_value = 0
    mock_repo.delete_features.return_value = True
    return mock_repo


@pytest.fixture
def mock_feature_view_name_mapper() -> Mock:
    """Create a mock FeatureViewNameMapper for testing."""
    mock_mapper = Mock()
    mock_mapper.map.return_value = "test_feature_view"
    return mock_mapper


@pytest.fixture
def feature_version_info() -> FeatureConfigVersionInfo:
    """Create a test feature configuration version info."""
    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="abc123def456",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[{"name": "test_feature", "enabled": True}]
    )


@pytest.fixture
def feature_store_save_repository(
    feature_store_config: FeatureStoreConfig,
    mock_feast_provider: Mock,
    mock_offline_repo: Mock,
    mock_feature_view_name_mapper: Mock
) -> FeatureStoreSaveRepository:
    """Create a FeatureStoreSaveRepository for testing with mocked dependencies."""
    return FeatureStoreSaveRepository(
        config=feature_store_config,
        feast_provider=mock_feast_provider,
        offline_repo=mock_offline_repo,
        feature_view_name_mapper=mock_feature_view_name_mapper
    )
