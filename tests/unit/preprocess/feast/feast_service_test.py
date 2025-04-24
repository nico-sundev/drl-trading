from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from feast import Entity
from pandas import DataFrame

from ai_trading.config.feature_config import FeatureStoreConfig
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import FeastService


@pytest.fixture
def mock_feature_store_config() -> FeatureStoreConfig:
    """Create a mock feature store configuration."""
    return FeatureStoreConfig(
        enabled=True,
        repo_path="test/path",
        offline_store_path="test/offline/path",
        entity_name="test_entity",
        ttl_days=7,
        online_enabled=True,
    )


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
def mock_feast_service(mock_feature_store_config, mock_asset_data) -> FeastService:
    """Create a FeastService instance with mocked dependencies."""
    # Patch where FeatureStore is looked up
    with patch(
        "ai_trading.preprocess.feast.feast_service.FeatureStore"
    ) as mock_feast_constructor:
        mock_store_instance = MagicMock()
        mock_feast_constructor.return_value = mock_store_instance

        # Instantiate the service (constructor will use the mock)
        service = FeastService(
            feature_store_config=mock_feature_store_config,
            symbol="EURUSD",
            asset_data=mock_asset_data,
        )
        # Explicitly set mocks on the instance for tests using this fixture
        service.feature_store = MagicMock()
        service.entity = MagicMock()
        return service


def test_init_with_enabled_config(
    mock_feature_store_config: FeatureStoreConfig, mock_asset_data: AssetPriceDataSet
) -> None:
    """Test FeastService initialization with enabled config."""
    # Given
    # Patch where FeatureStore is looked up
    with patch("ai_trading.preprocess.feast.feast_service.FeatureStore") as mock_feast:
        mock_store = MagicMock()
        mock_feast.return_value = mock_store

        # When
        service = FeastService(
            feature_store_config=mock_feature_store_config,
            symbol="EURUSD",
            asset_data=mock_asset_data,
        )

        # Then
        # Check if the service attribute holds the mock instance
        assert service.feature_store is mock_store
        assert service.entity is not None  # Entity is still initialized normally
        # Check if the mock constructor was called correctly
        mock_feast.assert_called_once_with(
            repo_path=mock_feature_store_config.repo_path
        )


def test_init_with_disabled_config(
    mock_feature_store_config: FeatureStoreConfig, mock_asset_data: AssetPriceDataSet
) -> None:
    """Test FeastService initialization with disabled config."""
    # Given
    mock_feature_store_config.enabled = False

    # Patch where FeatureStore is looked up
    with patch("ai_trading.preprocess.feast.feast_service.FeatureStore") as mock_feast:
        # When
        service = FeastService(
            feature_store_config=mock_feature_store_config,
            symbol="EURUSD",
            asset_data=mock_asset_data,
        )

        # Then
        assert service.feature_store is None
        assert service.entity is None
        mock_feast.assert_not_called()


def test_get_entity_value(mock_feast_service):
    """Test get_entity_value returns correctly formatted entity ID."""
    # Given
    expected_entity_value = "EURUSD_H1"

    # When
    actual_entity_value = mock_feast_service.get_entity_value()

    # Then
    assert actual_entity_value == expected_entity_value


def test_get_feature_view_name(mock_feast_service):
    """Test get_feature_view_name returns correctly formatted view name."""
    # Given
    feature_name = "TestFeature"
    param_hash = "abc123"
    expected_view_name = "TestFeature_H1_abc123"

    # When
    actual_view_name = mock_feast_service.get_feature_view_name(
        feature_name, param_hash
    )

    # Then
    assert actual_view_name == expected_view_name


def test_is_enabled_when_enabled(mock_feast_service):
    """Test is_enabled returns True when service is enabled."""
    # Given
    mock_feast_service.config.enabled = True
    mock_feast_service.feature_store = MagicMock()

    # When
    result = mock_feast_service.is_enabled()

    # Then
    assert result is True


def test_is_enabled_when_disabled(mock_feast_service):
    """Test is_enabled returns False when service is disabled."""
    # Given
    mock_feast_service.config.enabled = False

    # When
    result = mock_feast_service.is_enabled()

    # Then
    assert result is False


def test_get_historical_features_when_found(mock_feast_service, mock_asset_data):
    """Test get_historical_features returns data when found in store."""
    # Given
    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]

    expected_df = DataFrame({"col1": [1, 2, 3]})
    mock_response = MagicMock()
    mock_response.to_df.return_value = expected_df

    mock_feast_service.feature_store.get_historical_features.return_value = (
        mock_response
    )

    # When
    result = mock_feast_service.get_historical_features(
        feature_name=feature_name,
        param_hash=param_hash,
        sub_feature_names=sub_feature_names,
    )

    # Then
    assert result is expected_df
    mock_feast_service.feature_store.get_historical_features.assert_called_once()


def test_get_historical_features_when_not_found(mock_feast_service, mock_asset_data):
    """Test get_historical_features returns None when not found in store."""
    # Given
    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]

    mock_response = MagicMock()
    mock_response.to_df.return_value = DataFrame()

    mock_feast_service.feature_store.get_historical_features.return_value = (
        mock_response
    )

    # When
    result = mock_feast_service.get_historical_features(
        feature_name=feature_name,
        param_hash=param_hash,
        sub_feature_names=sub_feature_names,
    )

    # Then
    assert result is None


def test_get_historical_features_when_exception(mock_feast_service, mock_asset_data):
    """Test get_historical_features gracefully handles exceptions."""
    # Given
    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]

    mock_feast_service.feature_store.get_historical_features.side_effect = Exception(
        "Test error"
    )

    # When
    result = mock_feast_service.get_historical_features(
        feature_name=feature_name,
        param_hash=param_hash,
        sub_feature_names=sub_feature_names,
    )

    # Then
    assert result is None


def test_create_feature_view(mock_feast_service):
    """Test _create_feature_view creates a proper feature view."""
    # Given
    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]
    source_path = "test/path/file.parquet"

    # Use a real Entity for this test as it's simple and needed by the method
    mock_feast_service.entity = Entity(name="test_entity", join_keys=["test_entity"])

    # When
    # Patch where FileSource and FeatureView are looked up
    with patch(
        "ai_trading.preprocess.feast.feast_service.FileSource"
    ) as mock_file_source:
        with patch(
            "ai_trading.preprocess.feast.feast_service.FeatureView"
        ) as mock_feature_view:
            mock_source_instance = MagicMock()
            mock_view_instance = MagicMock()
            mock_file_source.return_value = mock_source_instance
            mock_feature_view.return_value = mock_view_instance

            result = mock_feast_service._create_feature_view(
                feature_name=feature_name,
                param_hash=param_hash,
                sub_feature_names=sub_feature_names,
                source_path=source_path,
            )

            # Then
            assert (
                result is mock_view_instance
            )  # Ensure the mocked view instance is returned

            # Check FileSource call
            feature_view_name = mock_feast_service.get_feature_view_name(
                feature_name, param_hash
            )
            mock_file_source.assert_called_once_with(
                name=f"{feature_view_name}_source",
                path=source_path,
                timestamp_field="event_timestamp",
            )

            # Check FeatureView call (basic check, can add more specific arg checks if needed)
            mock_feature_view.assert_called_once()
            # Example of checking specific args (adjust as needed)
            args, kwargs = mock_feature_view.call_args
            assert kwargs["name"] == feature_view_name
            assert kwargs["source"] is mock_source_instance
            assert "H1" in feature_view_name  # Check timeframe inclusion remains


def test_store_computed_features_success(
    mock_feast_service: FeastService, mock_asset_data: AssetPriceDataSet
) -> None:
    """Test store_computed_features correctly stores features."""
    # Given
    feature_df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=5, freq="H"),
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]

    mock_feast_service.feature_store.get_feature_view.side_effect = Exception(
        "Not found"
    )

    # When
    with patch("pandas.DataFrame.to_parquet"):
        mock_feast_service._create_feature_view = MagicMock()
        mock_feast_service._create_feature_view.return_value = MagicMock()

        mock_feast_service.store_computed_features(
            feature_df=feature_df,
            feature_name=feature_name,
            param_hash=param_hash,
            sub_feature_names=sub_feature_names,
        )

        # Then
        mock_feast_service.feature_store.apply.assert_called_once()
        mock_feast_service._create_feature_view.assert_called_once()


def test_store_computed_features_with_exception(
    mock_feast_service: FeastService, mock_asset_data: AssetPriceDataSet
) -> None:
    """Test store_computed_features gracefully handles exceptions."""
    # Given
    feature_df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=5, freq="H"),
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    feature_name = "TestFeature"
    param_hash = "abc123"
    sub_feature_names = ["feature1", "feature2"]

    mock_feast_service.feature_store.get_feature_view.side_effect = Exception(
        "Not found"
    )

    # When
    with patch("pandas.DataFrame.to_parquet") as mock_to_parquet:
        mock_to_parquet.side_effect = Exception("Failed to save")

        # Test should not raise exception but log warning
        mock_feast_service.store_computed_features(
            feature_df=feature_df,
            feature_name=feature_name,
            param_hash=param_hash,
            sub_feature_names=sub_feature_names,
        )

        # Then no assertion needed as we're testing it doesn't crash
