"""
Test fixtures for offline_feature_local_repo tests.

Provides fixtures and utilities for testing the OfflineFeatureLocalRepo class.
"""

import tempfile
from typing import Generator

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo import (
    OfflineFeatureLocalRepo,
)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def feature_store_config(temp_dir: str) -> FeatureStoreConfig:
    """Create a feature store configuration for testing."""
    return FeatureStoreConfig(
        enabled=True,
        repo_path=temp_dir,
        offline_store_path=temp_dir,  # Use temp_dir as offline store path
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test_service",
        service_version="1.0.0"
    )


@pytest.fixture
def offline_repo(feature_store_config: FeatureStoreConfig) -> OfflineFeatureLocalRepo:
    """Create an OfflineFeatureLocalRepo instance for testing."""
    return OfflineFeatureLocalRepo(feature_store_config)


@pytest.fixture
def sample_features_df() -> DataFrame:
    """Create sample features DataFrame with realistic data."""
    return DataFrame({
        "event_timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ],
        "feature_1": [1.5, 2.5, 3.5],
        "feature_2": [10.0, 20.0, 30.0],
        "rsi_14": [30.5, 45.2, 67.8]
    })


@pytest.fixture
def eurusd_h1_symbol() -> str:
    """Create a sample symbol for EUR/USD H1."""
    return "EURUSD"


@pytest.fixture
def gbpusd_m15_symbol() -> str:
    """Create a sample symbol for GBP/USD M15."""
    return "GBPUSD"
