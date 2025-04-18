from datetime import datetime

import pytest
from pandas import DataFrame

from ai_trading.config.feature_config import FeaturesConfig, FeatureStoreConfig
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


@pytest.fixture
def feature_aggregator():
    source_df = DataFrame(
        {
            "Time": [datetime.now()],
            "Open": [1.0],
            "High": [1.1],
            "Low": [0.9],
            "Close": [1.0],
            "Volume": [1000],
        }
    )
    config = FeaturesConfig(feature_definitions=[])
    registry = FeatureClassRegistry()
    return FeatureAggregator(source_df, config, registry)


@pytest.fixture
def feature_aggregator_with_symbol():
    source_df = DataFrame(
        {
            "Time": [datetime.now()],
            "Open": [1.0],
            "High": [1.1],
            "Low": [0.9],
            "Close": [1.0],
            "Volume": [1000],
            "symbol": ["GBPUSD"],
        }
    )
    config = FeaturesConfig(feature_definitions=[])
    registry = FeatureClassRegistry()
    return FeatureAggregator(source_df, config, registry)


def test_get_symbol_from_df_default(feature_aggregator):
    """Test that _get_symbol_from_df returns default when no symbol column exists"""
    assert feature_aggregator._get_symbol_from_df() == "EURUSD"


def test_get_symbol_from_df_with_symbol(feature_aggregator_with_symbol):
    """Test that _get_symbol_from_df returns correct symbol when symbol column exists"""
    assert feature_aggregator_with_symbol._get_symbol_from_df() == "GBPUSD"


def test_get_symbol_from_df_with_store_config(feature_aggregator_with_symbol):
    """Test that feature store is properly initialized with symbol config"""
    store_config = FeatureStoreConfig(
        enabled=True,
        repo_path="test_repo",
        offline_store_path="test_store.parquet",
        entity_name="symbol",
        ttl_days=1,
        online_enabled=True,
    )
    aggregator = FeatureAggregator(
        feature_aggregator_with_symbol.source_df,
        feature_aggregator_with_symbol.config,
        feature_aggregator_with_symbol.class_registry,
        store_config,
    )
    assert aggregator._get_symbol_from_df() == "GBPUSD"
