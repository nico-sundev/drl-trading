"""
Test fixtures for feature_store repository tests.

Provides common test fixtures and mocks for testing the FeatureStore repository classes.
"""

import tempfile
from datetime import datetime
from typing import Generator
from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig, LocalRepoConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from types import SimpleNamespace
from drl_trading_common.adapter.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.dto.feature_service_metadata import (
    FeatureServiceMetadata,
)
from drl_trading_core.core.dto.feature_view_metadata import FeatureViewMetadata
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier
from feast import FeatureService, FeatureStore
from pandas import DataFrame

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def feature_store_config(temp_dir: str) -> FeatureStoreConfig:
    """Create a test feature store configuration."""
    local_repo_config = LocalRepoConfig(repo_path=temp_dir)

    return FeatureStoreConfig(
        cache_enabled=True,
        config_directory=temp_dir,
        entity_name="trading_entity",
        ttl_days=30,
        online_enabled=True,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
        local_repo_config=local_repo_config
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
def mock_feast_provider() -> Mock:
    """Create a mock FeastProvider for testing."""
    mock_provider = Mock()

    # Create mock feature store with proper schema mocking
    mock_feature_store = Mock()

    # Mock feature view with schema
    mock_feature_view = Mock()
    mock_schema_field = Mock()
    mock_schema_field.name = "feature_1"
    mock_feature_view.schema = [mock_schema_field]  # Make schema iterable

    mock_feature_store.get_feature_view.return_value = mock_feature_view
    mock_feature_store.materialize.return_value = None
    mock_feature_store.write_to_online_store.return_value = None
    mock_feature_store.apply.return_value = None

    mock_provider.get_feature_store.return_value = mock_feature_store
    # Updated provider API to align with repository implementation
    default_fv = SimpleNamespace(name="test_feature_view")
    mock_provider.create_feature_view_from_request.return_value = default_fv
    feature_service_mock = Mock()
    feature_service_mock.name = "test_service"
    mock_provider.create_feature_service.return_value = feature_service_mock
    entity_mock = Mock()
    entity_mock.name = "trading_entity"
    mock_provider.get_entity.return_value = entity_mock
    return mock_provider


@pytest.fixture
def mock_feature_store() -> Mock:
    """Create a mock FeatureStore for testing."""
    mock_store = Mock(spec=FeatureStore)

    # Mock online features response
    mock_online_response = Mock()
    mock_online_response.to_df.return_value = DataFrame({
        "symbol": ["EURUSD"],
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
def feature_service_request(
    eurusd_h1_symbol: str,
    feature_version_info: FeatureConfigVersionInfo,
) -> FeatureServiceMetadata:
    """Create a test FeatureServiceRequestContainer."""
    dataset_identifier = DatasetIdentifier(symbol=eurusd_h1_symbol, timeframe=Timeframe.HOUR_1)
    return FeatureServiceMetadata.create(
        dataset_identifier=dataset_identifier,
        feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
        feature_config_version=feature_version_info,
        feature_view_metadata_list=[]
    )


@pytest.fixture
def gbpusd_feature_service_request(
    gbpusd_m15_symbol: str,
    feature_version_info: FeatureConfigVersionInfo,
) -> FeatureServiceMetadata:
    """Create a test FeatureServiceRequestContainer for GBPUSD."""
    dataset_identifier = DatasetIdentifier(symbol=gbpusd_m15_symbol, timeframe=Timeframe.MINUTE_15)
    return FeatureServiceMetadata.create(
        dataset_identifier=dataset_identifier,
        feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
        feature_config_version=feature_version_info,
        feature_view_metadata_list=[]
    )


@pytest.fixture
def feature_view_requests(
    eurusd_h1_symbol: str,
) -> list[FeatureViewMetadata]:
    """Create sample FeatureViewRequest list for tests."""
    from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier
    dataset_identifier = DatasetIdentifier(symbol=eurusd_h1_symbol, timeframe=Timeframe.HOUR_1)
    return [
        FeatureViewMetadata(
            dataset_identifier=dataset_identifier,
            feature_metadata=Mock(),  # Mock feature metadata
        ),
        FeatureViewMetadata(
            dataset_identifier=dataset_identifier,
            feature_metadata=Mock(),
        ),
    ]

@pytest.fixture
def eurusd_h1_symbol() -> str:
    """Create a sample symbol for EUR/USD H1."""
    return "EURUSD"


@pytest.fixture
def gbpusd_m15_symbol() -> str:
    """Create a sample symbol for GBP/USD M15."""
    return "GBPUSD"
