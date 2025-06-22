"""
Pytest fixtures for offline feature repository tests.

Provides common test fixtures for feature storage testing including
configurations, sample data, and temporary repositories.
"""

import tempfile
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
from pandas import DataFrame

from drl_trading_core.preprocess.feast.feature_store_save_repo import (
    FeatureStoreSaveRepo,
)
from drl_trading_core.preprocess.feast.offline_feature_local_repo import (
    OfflineFeatureLocalRepo,
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
        offline_store_path=temp_dir,
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test_service",
        service_version="1.0.0"
    )


@pytest.fixture
def offline_repo(feature_store_config: FeatureStoreConfig) -> OfflineFeatureLocalRepo:
    """Create an offline feature repository for testing."""
    return OfflineFeatureLocalRepo(feature_store_config)


@pytest.fixture
def sample_features_df() -> DataFrame:
    """Create sample features DataFrame for testing."""
    return DataFrame({
        "event_timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ],
        "feature_1": [1.5, 2.5, 3.5],
        "feature_2": [10.0, 20.0, 30.0],
        "feature_3": [100, 200, 300]
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
    mock_feature_store = Mock()
    mock_provider.get_feature_store.return_value = mock_feature_store

    # Mock feature view creation
    mock_obs_fv = Mock()
    mock_reward_fv = Mock()
    mock_provider.create_feature_view.side_effect = [mock_obs_fv, mock_reward_fv]

    # Mock feature service creation
    mock_feature_service = Mock()
    mock_feature_service.name = "test_feature_service"
    mock_provider.create_feature_service.return_value = mock_feature_service

    return mock_provider


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
def feature_version_info() -> FeatureConfigVersionInfo:
    """Create a test feature configuration version info."""
    from datetime import datetime

    return FeatureConfigVersionInfo(
        semver="1.0.0",
        hash="abc123def456",
        created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        feature_definitions=[{"name": "test_feature", "enabled": True}]
    )


@pytest.fixture
def feature_store_save_repo(
    feature_store_config: FeatureStoreConfig,
    mock_feast_provider: Mock,
    mock_offline_repo: Mock
) -> FeatureStoreSaveRepo:
    """Create a FeatureStoreSaveRepo for testing with mocked dependencies."""
    return FeatureStoreSaveRepo(
        config=feature_store_config,
        feast_provider=mock_feast_provider,
        offline_repo=mock_offline_repo
    )
