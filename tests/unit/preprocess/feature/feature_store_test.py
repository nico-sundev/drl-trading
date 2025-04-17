import os
import pytest
from unittest.mock import MagicMock, patch
from pandas import DataFrame, date_range
import numpy as np

from ai_trading.config.feature_config import FeatureStoreConfig, FeaturesConfig, FeatureDefinition
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.feature_repo.feature_store_service import FeatureStoreService


@pytest.fixture
def mock_data() -> DataFrame:
    return DataFrame({
        "Time": date_range(start="2022-01-01", periods=5, freq="D"),
        "Open": [1, 2, 3, 4, 5],
        "High": [2, 3, 4, 5, 6],
        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5]
    })

@pytest.fixture
def feature_store_config():
    return FeatureStoreConfig(
        enabled=True,
        repo_path="test_feature_repo",
        offline_store_path="test_data/features.parquet",
        entity_name="symbol",
        ttl_days=1,
        online_enabled=True
    )

@pytest.fixture
def features_config():
    return FeaturesConfig(
        feature_definitions=[
            FeatureDefinition(
                name="rsi",
                enabled=True,
                derivatives=[],
                parameter_sets=[{"enabled": True, "length": 14, "type": "rsi"}]
            )
        ]
    )

@pytest.fixture
def class_registry():
    return FeatureClassRegistry()

def test_feature_store_service_creation(feature_store_config, features_config, class_registry):
    # Given
    service = FeatureStoreService(
        config=features_config,
        feature_store_config=feature_store_config,
        class_registry=class_registry
    )

    # When
    feature_views = service.create_feature_views()

    # Then
    assert len(feature_views) > 0
    feature_view = feature_views[0]
    assert feature_view.name.startswith("rsi_")
    assert feature_view.entities[0].name == "symbol"
    assert feature_view.ttl.days == 1
    assert feature_view.online
    assert "feature_type" in feature_view.tags
    assert feature_view.tags["feature_type"] == "rsi"

@patch("feast.FeatureStore")
def test_feature_aggregator_with_store(mock_feast, mock_data, feature_store_config, features_config, class_registry):
    # Given
    from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
    mock_store = MagicMock()
    mock_feast.return_value = mock_store
    
    # Mock the get_historical_features to return None, forcing computation
    mock_store.get_historical_features.return_value = MagicMock()
    mock_store.get_historical_features.return_value.to_df.return_value = DataFrame()
    
    aggregator = FeatureAggregator(
        source_df=mock_data,
        config=features_config,
        class_registry=class_registry,
        feature_store_config=feature_store_config
    )

    # When
    result = aggregator.compute()

    # Then
    assert not result.empty
    assert "Time" in result.columns
    assert any(col.startswith("rsi_14") for col in result.columns)
    mock_store.get_historical_features.assert_called()