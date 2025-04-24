"""Unit tests for the FeatureAggregator class."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from pandas import DataFrame

from ai_trading.config.feature_config import FeatureDefinition, FeaturesConfig
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import FeastService
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def compute(self) -> DataFrame:
        """Generate mock feature data."""
        df = self.df_source.copy()
        df["feature1"] = 1.0
        df["feature2"] = 2.0
        return df

    def get_sub_features_names(self) -> list[str]:
        """Return mock sub-feature names."""
        return ["feature1", "feature2"]


@pytest.fixture
def mock_param_set() -> MagicMock:
    """Create a mock parameter set for features."""
    param_set = MagicMock()
    param_set.enabled = True
    param_set.hash_id.return_value = "abc123"
    return param_set


@pytest.fixture
def mock_feature_definition(mock_param_set) -> FeatureDefinition:
    """Create a mock feature definition."""
    return FeatureDefinition(
        name="MockFeature",
        enabled=True,
        derivatives=[],
        parameter_sets=[],
        parsed_parameter_sets=[mock_param_set],
    )


@pytest.fixture
def mock_features_config(mock_feature_definition) -> FeaturesConfig:
    """Create a mock features configuration."""
    return FeaturesConfig(feature_definitions=[mock_feature_definition])


@pytest.fixture
def mock_asset_data() -> AssetPriceDataSet:
    """Create a mock asset price dataset."""
    df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            "Open": [1.0] * 10,
            "High": [2.0] * 10,
            "Low": [0.5] * 10,
            "Close": [1.5] * 10,
            "Volume": [1000.0] * 10,
        }
    )

    return AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=df,
    )


@pytest.fixture
def mock_class_registry() -> FeatureClassRegistry:
    """Create a mock feature class registry."""
    registry = MagicMock()
    registry.feature_class_map = {"MockFeature": MockFeature}
    return registry


@pytest.fixture
def mock_feast_service() -> FeastService:
    """Create a mock FeastService."""
    feast_service = MagicMock(spec=FeastService)
    feast_service.is_enabled.return_value = True
    feast_service.get_historical_features.return_value = None
    return feast_service


@pytest.fixture
def feature_aggregator(
    mock_asset_data, mock_features_config, mock_class_registry, mock_feast_service
) -> FeatureAggregator:
    """Create a FeatureAggregator instance with mocked dependencies."""
    return FeatureAggregator(
        asset_data=mock_asset_data,
        symbol="EURUSD",
        config=mock_features_config,
        class_registry=mock_class_registry,
        feast_service=mock_feast_service,
    )


def test_compute_with_no_features(feature_aggregator):
    """Test compute returns empty DataFrame when no features are enabled."""
    # Given
    feature_aggregator.config.feature_definitions[0].enabled = False

    # When
    result = feature_aggregator.compute()

    # Then
    assert result.empty


def test_compute_with_disabled_parameter_sets(feature_aggregator):
    """Test compute returns empty DataFrame when no parameter sets are enabled."""
    # Given
    feature_aggregator.config.feature_definitions[0].parsed_parameter_sets[
        0
    ].enabled = False

    # When
    result = feature_aggregator.compute()

    # Then
    assert result.empty


def test_compute_with_cached_features(feature_aggregator, mock_feast_service):
    """Test compute uses cached features from feature store when available."""
    # Given
    historical_features = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
        }
    )
    mock_feast_service.get_historical_features.return_value = historical_features

    # When
    result = feature_aggregator.compute()

    # Then
    assert not result.empty
    mock_feast_service.get_historical_features.assert_called_once()
    mock_feast_service.store_computed_features.assert_not_called()


def test_compute_without_cached_features(feature_aggregator, mock_feast_service):
    """Test compute calculates and stores features when not in feature store."""
    # Given
    mock_feast_service.get_historical_features.return_value = None

    # When
    result = feature_aggregator.compute()

    # Then
    assert not result.empty
    mock_feast_service.get_historical_features.assert_called_once()
    mock_feast_service.store_computed_features.assert_called_once()

    # Verify computed features contain expected columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns


def test_compute_with_disabled_feature_store(feature_aggregator, mock_feast_service):
    """Test compute works when feature store is disabled."""
    # Given
    mock_feast_service.is_enabled.return_value = False

    # When
    result = feature_aggregator.compute()

    # Then
    assert not result.empty
    mock_feast_service.get_historical_features.assert_called_once()
    mock_feast_service.store_computed_features.assert_not_called()

    # Verify computed features contain expected columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns


def test_compute_handles_multiple_feature_results(
    feature_aggregator, mock_feast_service, mock_features_config
):
    """Test compute correctly combines multiple feature results."""
    # Given
    # Add second feature definition
    mock_param_set2 = MagicMock()
    mock_param_set2.enabled = True
    mock_param_set2.hash_id.return_value = "def456"

    mock_feature_def2 = FeatureDefinition(
        name="MockFeature",
        enabled=True,
        derivatives=[],
        parameter_sets=[],
        parsed_parameter_sets=[mock_param_set2],
    )

    feature_aggregator.config.feature_definitions.append(mock_feature_def2)

    # Configure first feature to come from cache
    cached_features = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            "cached_feature1": [10.0] * 10,
            "cached_feature2": [20.0] * 10,
        }
    )

    # Mock to return cached results only for first feature
    def mock_get_historical_features(feature_name, param_hash, sub_feature_names):
        if param_hash == "abc123":
            return cached_features
        return None

    mock_feast_service.get_historical_features.side_effect = (
        mock_get_historical_features
    )

    # When
    result = feature_aggregator.compute()

    # Then
    assert not result.empty
    # Should be called twice (once for each feature)
    assert mock_feast_service.get_historical_features.call_count == 2
    # Should be called once (only for the second feature that wasn't cached)
    assert mock_feast_service.store_computed_features.call_count == 1

    # Check that result contains columns from both features
    assert "cached_feature1" in result.columns or "feature1" in result.columns
    assert "cached_feature2" in result.columns or "feature2" in result.columns
