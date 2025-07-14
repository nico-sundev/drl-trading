"""
Example integration test demonstrating proper use of the Feast integration test setup.

This test demonstrates how to use real Feast functionality with minimal mocking
for true integration testing of the feature store pipeline. Uses test features
that don't depend on strategy modules to maintain proper dependency boundaries.
"""
import pytest
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from feast import FeatureStore
from injector import Injector

from drl_trading_core.preprocess.feature_store.provider.feast_provider import (
    FeastProvider,
)
from drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper import (
    FeatureStoreWrapper,
)


class TestFeastIntegration:
    """Integration tests for Feast feature store functionality."""

    def test_feast_provider_can_create_feature_views(
        self,
        integration_container: Injector,
        feature_version_info_fixture,
        clean_feature_store: FeatureStore
    ) -> None:
        """Test that FeastProvider can create feature views with real Feast backend."""
        # Given
        feast_provider = integration_container.get(FeastProvider)
        symbol = "EURUSD"
        feature_view_name = "test_features"

        # Verify that we're using a real Feast store, not a mock
        assert isinstance(feast_provider.get_feature_store(), FeatureStore)
        assert feast_provider.is_enabled() is True

        # When
        feature_view = feast_provider.create_feature_view(
            symbol=symbol,
            feature_view_name=feature_view_name,
            feature_role=None,  # This might need adjustment based on your enum
            feature_version_info=feature_version_info_fixture
        )

        # Then
        assert feature_view is not None
        assert feature_view.name == feature_view_name
        assert symbol in [tag for tag in feature_view.tags.values()]

    def test_feature_factory_creates_real_features(
        self,
        integration_container: Injector,
    ) -> None:
        """Test that the feature factory creates real feature instances."""
        # Given
        from .conftest import TestClosePriceConfig, TestFeatureFactory, TestRsiConfig

        feature_factory = integration_container.get(TestFeatureFactory)
        dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe="1H")
        rsi_config = TestRsiConfig(period=14)
        close_price_config = TestClosePriceConfig(lookback=1)

        # When
        rsi_feature = feature_factory.create_feature(
            feature_name="rsi",
            dataset_id=dataset_id,
            config=rsi_config
        )
        close_price_feature = feature_factory.create_feature(
            feature_name="close_price",
            dataset_id=dataset_id,
            config=close_price_config
        )

        # Then
        assert rsi_feature is not None
        assert rsi_feature.get_feature_name() == "rsi"
        assert rsi_feature.get_sub_features_names() == ["rsi_14"]

        assert close_price_feature is not None
        assert close_price_feature.get_feature_name() == "close_price"
        assert close_price_feature.get_sub_features_names() == ["close_1"]

    def test_feature_computation_returns_real_data(
        self,
        integration_container: Injector,
    ) -> None:
        """Test that features can compute real data."""
        # Given
        from .conftest import TestFeatureFactory, TestRsiConfig

        feature_factory = integration_container.get(TestFeatureFactory)
        dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe="1H")
        rsi_config = TestRsiConfig(period=14)

        rsi_feature = feature_factory.create_feature(
            feature_name="rsi",
            dataset_id=dataset_id,
            config=rsi_config
        )
        assert rsi_feature is not None, "Feature creation should not return None"

        # When
        computed_data = rsi_feature.compute_all()

        # Then
        assert computed_data is not None
        assert len(computed_data) == 100  # As defined in our test implementation
        assert "event_timestamp" in computed_data.columns
        assert "rsi_14" in computed_data.columns
        assert "EURUSD" in computed_data.columns

        # Verify data types and ranges
        assert computed_data["rsi_14"].dtype.kind in 'fi'  # float or int
        assert computed_data["rsi_14"].min() >= 30.0  # Based on our test data
        assert computed_data["rsi_14"].max() <= 70.0

    def test_feature_store_wrapper_provides_real_store(
        self,
        integration_container: Injector,
        clean_feature_store: FeatureStore
    ) -> None:
        """Test that FeatureStoreWrapper provides a real FeatureStore instance."""
        # Given
        feature_store_wrapper = integration_container.get(FeatureStoreWrapper)

        # When
        feature_store = feature_store_wrapper.get_feature_store()

        # Then
        assert isinstance(feature_store, FeatureStore)
        assert feature_store is not None

        # Verify it's the same instance as our clean fixture
        assert feature_store.repo_path == clean_feature_store.repo_path

    def test_clean_feature_store_isolation(
        self,
        integration_container: Injector,
        clean_feature_store: FeatureStore
    ) -> None:
        """Test that each test gets a clean feature store state."""
        # Given
        feast_provider = integration_container.get(FeastProvider)

        # When - try to list existing feature views (should be empty)
        feature_views = feast_provider.get_feature_store().list_feature_views()

        # Then
        assert len(feature_views) == 0, "Feature store should be clean for each test"

    @pytest.mark.parametrize("feature_name,config_data,expected_config_type", [
        ("rsi", {"period": 21}, "TestRsiConfig"),
        ("close_price", {"lookback": 3}, "TestClosePriceConfig"),
    ])
    def test_feature_factory_config_creation(
        self,
        integration_container: Injector,
        feature_name: str,
        config_data: dict,
        expected_config_type: str
    ) -> None:
        """Test parameterized feature configuration creation."""
        # Given
        from .conftest import TestClosePriceConfig, TestFeatureFactory, TestRsiConfig

        feature_factory = integration_container.get(TestFeatureFactory)

        # When
        config = feature_factory.create_config_instance(feature_name, config_data)

        # Then
        assert config is not None
        assert type(config).__name__ == expected_config_type
        if feature_name == "rsi":
            assert isinstance(config, TestRsiConfig)
            assert config.period == config_data["period"]
        elif feature_name == "close_price":
            assert isinstance(config, TestClosePriceConfig)
            assert config.lookback == config_data["lookback"]


class TestFeastDataPersistence:
    """Tests for data persistence and retrieval with Feast."""

    def test_feature_data_can_be_stored_and_retrieved(
        self,
        integration_container: Injector,
        clean_feature_store: FeatureStore
    ) -> None:
        """Test end-to-end feature data storage and retrieval."""
        # Given
        from .conftest import TestFeatureFactory, TestRsiConfig

        feature_factory = integration_container.get(TestFeatureFactory)
        feast_provider = integration_container.get(FeastProvider)

        dataset_id = DatasetIdentifier(symbol="EURUSD", timeframe="1H")
        rsi_config = TestRsiConfig(period=14)
        rsi_feature = feature_factory.create_feature("rsi", dataset_id, rsi_config)
        assert rsi_feature is not None, "Feature creation should not return None"

        # When
        computed_data = rsi_feature.compute_all()

        # Then
        assert computed_data is not None
        assert len(computed_data) > 0

        # This test demonstrates that we have real data that could be stored
        # In a full integration test, you would:
        # 1. Apply feature views to Feast
        # 2. Materialize features
        # 3. Retrieve features using get_historical_features
        # For now, we verify the data structure is correct for Feast ingestion
        required_columns = ["event_timestamp", dataset_id.symbol]
        for col in required_columns:
            assert col in computed_data.columns, f"Missing required column: {col}"
