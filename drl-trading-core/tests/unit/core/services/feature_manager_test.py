"""Unit tests for the FeatureManager class."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dask.delayed import Delayed
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureDefinition, FeaturesConfig
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.timeframe import Timeframe
from pandas import DataFrame

from drl_trading_core.core.service.feature_manager import FeatureKey, FeatureManager

class MockFeature(BaseFeature):
    """Mock feature class for testing FeatureManager."""

    def __init__(self, feature_name: str = "MockFeature"):
        self.feature_name = feature_name
        self.data = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    def compute(self) -> DataFrame:
        """Return mock feature data."""
        return self.data.copy()

    def compute_all(self) -> DataFrame:
        """Return all mock feature data."""
        return self.data.copy()

    def compute_latest(self) -> DataFrame:
        """Return latest mock feature data."""
        return self.data.tail(1).copy()

    def add(self, df: DataFrame) -> None:
        """Add new data to mock feature."""
        self.data = pd.concat([self.data, df], ignore_index=True)

    def get_sub_features_names(self) -> list[str]:
        """Return mock sub-feature names."""
        return ["mock_feature"]

    def get_feature_name(self) -> str:
        """Return the base name of the feature."""
        return self.feature_name

    def get_config_to_string(self) -> str:
        return "Test"


@pytest.fixture
def mock_feature_factory() -> MagicMock:
    """Create a mock feature factory."""
    factory = MagicMock(spec=IFeatureFactory)
    factory.create_feature.return_value = MockFeature()
    return factory


@pytest.fixture
def mock_features_config() -> FeaturesConfig:
    """Create a mock features configuration."""
    # Create mock parameter set
    mock_param_set = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set.enabled = True
    mock_param_set.hash_id.return_value = "test_hash_123"
    mock_param_set.to_string.return_value = "test_params"
    mock_param_set.name = "test_param_set"

    # Create mock feature definition
    mock_feature_def = MagicMock(spec=FeatureDefinition)
    mock_feature_def.name = "TestFeature"
    mock_feature_def.enabled = True
    mock_feature_def.parsed_parameter_sets = [mock_param_set]



    # Create features config
    config = MagicMock(spec=FeaturesConfig)
    config.feature_definitions = [mock_feature_def]
    config.dataset_definitions = {
        "BTCUSD": [Timeframe.MINUTE_1],
        "ETHUSD": [Timeframe.MINUTE_5],
    }

    return config


@pytest.fixture
def feature_manager(mock_features_config, mock_feature_factory) -> FeatureManager:
    """Create a FeatureManager instance for testing."""
    return FeatureManager(config=mock_features_config, feature_factory=mock_feature_factory)


@pytest.fixture
def sample_dataframe() -> DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [104.0, 105.0, 106.0],
        "volume": [1000, 1100, 1200]
    })


@pytest.fixture
def sample_dataset_id() -> DatasetIdentifier:
    """Create a sample DatasetIdentifier for testing."""
    return DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)


@pytest.fixture
def sample_dataset_id_2() -> DatasetIdentifier:
    """Create a second sample DatasetIdentifier for testing."""
    return DatasetIdentifier("ETHUSD", Timeframe.MINUTE_5)


def create_feature_key(feature_name: str, dataset_id: DatasetIdentifier, param_hash: str) -> FeatureKey:
    """Helper function to create FeatureKey instances."""
    return FeatureKey(
        feature_name=feature_name,
        dataset_id=dataset_id,
        param_hash=param_hash
    )


class TestFeatureManagerInitialization:
    """Test cases for FeatureManager initialization."""

    def test_init_stores_dependencies(self, mock_features_config, mock_feature_factory):
        """
        Given: Valid configuration and feature factory
        When: FeatureManager is initialized
        Then: Dependencies are stored correctly and features dict is empty
        """
        # When
        manager = FeatureManager(config=mock_features_config, feature_factory=mock_feature_factory)

        # Then
        assert manager.config == mock_features_config
        assert manager.feature_factory == mock_feature_factory
        assert manager._features == {}

    def test_init_with_dependency_injection(self):
        """
        Given: Mock dependencies
        When: FeatureManager is initialized with @inject decorator
        Then: Dependencies are properly injected
        """
        # Given
        mock_config = MagicMock(spec=FeaturesConfig)
        mock_config.feature_definitions = []
        mock_factory = MagicMock(spec=IFeatureFactory)

        # When
        manager = FeatureManager(config=mock_config, feature_factory=mock_factory)

        # Then
        assert manager.config is mock_config
        assert manager.feature_factory is mock_factory


class TestFeatureInitialization:
    """Test cases for feature initialization process."""

    def test_initialize_features_success(self, feature_manager):
        """
        Given: FeatureManager with valid configuration
        When: initialize_features is called
        Then: Features are created and stored correctly
        """
        # When
        feature_manager.initialize_features()

        # Then
        assert len(feature_manager._features) == 2  # 1 feature * 2 datasets
        assert feature_manager.feature_factory.create_feature.call_count == 2

    def test_generate_feature_configurations(self, feature_manager):
        """
        Given: FeatureManager with configuration containing enabled features and datasets
        When: _generate_feature_configurations is called
        Then: Correct number of configurations are generated
        """
        # When
        configurations = feature_manager._generate_feature_configurations()

        # Then
        assert len(configurations) == 2  # 1 feature * 2 datasets

        # Verify configuration structure
        feature_name, dataset_id, param_set = configurations[0]
        assert feature_name == "TestFeature"
        assert isinstance(dataset_id, DatasetIdentifier)
        assert param_set.enabled is True

    def test_generate_feature_configurations_with_disabled_feature(self, mock_features_config, mock_feature_factory):
        """
        Given: FeatureManager with disabled features
        When: _generate_feature_configurations is called
        Then: Disabled features are excluded from configurations
        """
        # Given
        mock_features_config.feature_definitions[0].enabled = False
        manager = FeatureManager(config=mock_features_config, feature_factory=mock_feature_factory)

        # When
        configurations = manager._generate_feature_configurations()

        # Then
        assert len(configurations) == 0

    def test_generate_feature_configurations_with_empty_parameter_sets(self, mock_feature_factory):
        """
        Given: FeatureManager with feature having empty parameter sets
        When: _generate_feature_configurations is called
        Then: Feature is included with None config
        """
        # Given
        mock_feature_def = MagicMock(spec=FeatureDefinition)
        mock_feature_def.name = "CloseFeature"
        mock_feature_def.enabled = True
        mock_feature_def.parameter_sets = []  # Empty parameter sets
        mock_feature_def.parsed_parameter_sets = []  # Empty parameter sets

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [mock_feature_def]
        config.dataset_definitions = {
            "BTCUSD": [Timeframe.MINUTE_1]
        }

        manager = FeatureManager(config=config, feature_factory=mock_feature_factory)

        # When
        configurations = manager._generate_feature_configurations()

        # Then
        assert len(configurations) == 1
        feature_name, dataset_id, param_set = configurations[0]
        assert feature_name == "CloseFeature"
        assert isinstance(dataset_id, DatasetIdentifier)
        assert param_set is None  # Should be None for features without config

    def test_create_features_batch_success(self, feature_manager):
        """
        Given: Valid feature configurations
        When: _create_features_batch is called
        Then: Features are created successfully
        """
        # Given
        configurations = feature_manager._generate_feature_configurations()
          # When
        created_features = feature_manager._create_features_batch(configurations)

        # Then
        assert len(created_features) == 2
        for feature_key, feature_instance in created_features:
            assert feature_key.feature_name == "TestFeature"
            assert feature_key.param_hash == "test_hash_123"
            assert isinstance(feature_instance, MockFeature)

    def test_create_features_batch_with_factory_failure(self, feature_manager):
        """
        Given: Feature factory that fails to create features
        When: _create_features_batch is called
        Then: Failed features are handled gracefully
        """
        # Given
        feature_manager.feature_factory.create_feature.return_value = None
        configurations = feature_manager._generate_feature_configurations()

        # When
        created_features = feature_manager._create_features_batch(configurations)

        # Then
        assert len(created_features) == 0

    def test_create_features_batch_with_exception(self, feature_manager):
        """
        Given: Feature factory that raises exceptions
        When: _create_features_batch is called
        Then: Exceptions are handled and logged
        """
        # Given
        feature_manager.feature_factory.create_feature.side_effect = Exception("Creation failed")
        configurations = feature_manager._generate_feature_configurations()

        # When
        created_features = feature_manager._create_features_batch(configurations)
          # Then
        assert len(created_features) == 0

    def test_store_features(self, feature_manager):
        """
        Given: List of created features
        When: _store_features is called
        Then: Features are stored in internal dictionary with correct keys
        """
        # Given
        mock_feature = MockFeature("TestFeature")
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash"
        )
        created_features = [(feature_key, mock_feature)]

        # When
        feature_manager._store_features(created_features)

        # Then
        assert feature_key in feature_manager._features
        assert feature_manager._features[feature_key] == mock_feature


class TestFeatureCreation:
    """Test cases for individual feature creation."""

    def test_create_feature_instance_success(self, feature_manager):
        """
        Given: Valid feature parameters
        When: _create_feature_instance is called
        Then: Feature instance is created successfully
        """
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        param_set = MagicMock(spec=BaseParameterSetConfig)

        # When
        feature = feature_manager._create_feature_instance("TestFeature", dataset_id, param_set)

        # Then
        assert feature is not None
        assert isinstance(feature, MockFeature)
        feature_manager.feature_factory.create_feature.assert_called_once_with(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            config=param_set,
            postfix=""
        )

    def test_create_feature_instance_with_postfix(self, feature_manager):
        """
        Given: Valid feature parameters with postfix
        When: _create_feature_instance is called with postfix
        Then: Feature instance is created with postfix
        """
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        param_set = MagicMock(spec=BaseParameterSetConfig)

        # When
        feature = feature_manager._create_feature_instance(
            "TestFeature", dataset_id, param_set, postfix="_custom"
        )

        # Then
        assert feature is not None
        feature_manager.feature_factory.create_feature.assert_called_once_with(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            config=param_set,
            postfix="_custom"
        )

    def test_create_feature_instance_without_config(self, feature_manager):
        """
        Given: Feature that doesn't require configuration
        When: _create_feature_instance is called with None config
        Then: Feature instance is created successfully without config
        """
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_manager.feature_factory.reset_mock()  # Reset the mock to clear previous calls

        # When
        feature = feature_manager._create_feature_instance("TestFeature", dataset_id, None)

        # Then
        assert feature is not None
        assert isinstance(feature, MockFeature)
        feature_manager.feature_factory.create_feature.assert_called_once_with(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            config=None,
            postfix=""
        )

    def test_create_feature_instance_factory_returns_none(self, feature_manager):
        """
        Given: Feature factory returns None
        When: _create_feature_instance is called
        Then: None is returned and error is logged
        """
        # Given
        feature_manager.feature_factory.create_feature.return_value = None
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        param_set = MagicMock(spec=BaseParameterSetConfig)

        # When
        feature = feature_manager._create_feature_instance("TestFeature", dataset_id, param_set)

        # Then
        assert feature is None

    def test_create_feature_instance_factory_raises_exception(self, feature_manager):
        """
        Given: Feature factory raises exception
        When: _create_feature_instance is called
        Then: None is returned and error is logged
        """
        # Given
        feature_manager.feature_factory.create_feature.side_effect = Exception("Factory error")
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        param_set = MagicMock(spec=BaseParameterSetConfig)

        # When
        feature = feature_manager._create_feature_instance("TestFeature", dataset_id, param_set)

        # Then
        assert feature is None


class TestFeatureRetrieval:
    """Test cases for feature retrieval operations."""

    def test_get_feature_existing(self, feature_manager):
        """
        Given: Feature exists in the manager
        When: get_feature is called with correct parameters
        Then: Feature instance is returned
        """
        # Given
        mock_feature = MockFeature("TestFeature")
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash"
        )
        feature_manager._features[feature_key] = mock_feature

        # When
        result = feature_manager.get_feature("TestFeature", dataset_id, "test_hash")

        # Then
        assert result == mock_feature

    def test_get_feature_non_existing(self, feature_manager):
        """
        Given: Feature does not exist in the manager
        When: get_feature is called
        Then: None is returned
        """
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.get_feature("NonExistentFeature", dataset_id, "invalid_hash")

        # Then
        assert result is None

    def test_get_all_features(self, feature_manager):
        """
        Given: Multiple features exist in the manager
        When: get_all_features is called
        Then: List of all feature instances is returned
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")
        dataset_id1 = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        dataset_id2 = DatasetIdentifier("ETHUSD", Timeframe.MINUTE_5)

        feature_key1 = FeatureKey(
            feature_name="Feature1",
            dataset_id=dataset_id1,
            param_hash="hash1"
        )
        feature_key2 = FeatureKey(
            feature_name="Feature2",
            dataset_id=dataset_id2,
            param_hash="hash2"
        )

        feature_manager._features[feature_key1] = feature1
        feature_manager._features[feature_key2] = feature2

        # When
        result = feature_manager.get_all_features()

        # Then
        assert len(result) == 2
        assert feature1 in result
        assert feature2 in result

    def test_get_all_features_empty(self, feature_manager) -> None:
        """
        Given: No features exist in the manager
        When: get_all_features is called
        Then: Empty list is returned
        """
        # When
        result = feature_manager.get_all_features()
        # Then
        assert result == []


class TestFeatureDataUpdates:
    """Test cases for updating feature data."""

    def test_update_features_data_success(self, feature_manager, sample_dataframe):
        """
        Given: Features exist in the manager
        When: update_features_data is called with new data
        Then: All features are updated with the new data
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")

        dataset_id1 = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        dataset_id2 = DatasetIdentifier("ETHUSD", Timeframe.MINUTE_5)

        feature_key1 = FeatureKey(feature_name="Feature1", dataset_id=dataset_id1, param_hash="hash1")
        feature_key2 = FeatureKey(feature_name="Feature2", dataset_id=dataset_id2, param_hash="hash2")

        feature_manager._features[feature_key1] = feature1
        feature_manager._features[feature_key2] = feature2

        # When
        feature_manager.update_features_data(sample_dataframe)

        # Then
        # Verify both features received the data (check their internal data was updated)
        assert len(feature1.data) == 6  # Original 3 + new 3
        assert len(feature2.data) == 6  # Original 3 + new 3    def test_update_features_data_with_exception(self, feature_manager, sample_dataframe):
        """
        Given: Feature that raises exception during update
        When: update_features_data is called
        Then: Exception is handled and other features are still updated
        """
        # Given
        good_feature = MockFeature("GoodFeature")
        bad_feature = MagicMock()
        bad_feature.add.side_effect = Exception("Update failed")

        dataset_id1 = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        dataset_id2 = DatasetIdentifier("ETHUSD", Timeframe.MINUTE_5)

        feature_key1 = FeatureKey(feature_name="GoodFeature", dataset_id=dataset_id1, param_hash="hash1")
        feature_key2 = FeatureKey(feature_name="BadFeature", dataset_id=dataset_id2, param_hash="hash2")

        feature_manager._features[feature_key1] = good_feature
        feature_manager._features[feature_key2] = bad_feature

        # When
        feature_manager.update_features_data(sample_dataframe)

        # Then
        # Good feature should still be updated
        assert len(good_feature.data) == 6
        # Bad feature's add method should have been called
        bad_feature.add.assert_called_once_with(sample_dataframe)

    def test_add_delegates_to_update_features_data(self, feature_manager, sample_dataframe):
        """
        Given: FeatureManager instance
        When: add method is called
        Then: It delegates to update_features_data
        """
        # Given
        with patch.object(feature_manager, 'update_features_data') as mock_update:
            # When
            feature_manager.add(sample_dataframe)

            # Then
            mock_update.assert_called_once_with(sample_dataframe)


class TestFeatureComputation:
    """Test cases for feature computation operations."""

    def test_compute_feature_success(self, feature_manager, sample_dataframe):
        """
        Given: Valid feature definition and parameter set
        When: compute_feature is called
        Then: Feature is computed and result is returned
        """
        # Given
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash"

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert result is not None
        assert isinstance(result, DataFrame)

    def test_compute_feature_disabled_feature(self, feature_manager, sample_dataframe):
        """
        Given: Disabled feature definition
        When: compute_feature is called
        Then: None is returned
        """
        # Given
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.enabled = False

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert result is None

    def test_compute_feature_disabled_param_set(self, feature_manager, sample_dataframe):
        """
        Given: Disabled parameter set
        When: compute_feature is called
        Then: None is returned
        """
        # Given
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.enabled = True

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = False

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)
          # Then
        assert result is None

    def test_compute_feature_uses_existing_instance(self, feature_manager, sample_dataframe):
        """
        Given: Feature instance already exists in manager
        When: compute_feature is called
        Then: Existing instance is used instead of creating new one
        """
        # Given
        existing_feature = MockFeature("TestFeature")
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash"
        )
        feature_manager._features[feature_key] = existing_feature

        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash"

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert result is not None
        # Factory should not be called since existing instance was used
        feature_manager.feature_factory.create_feature.assert_not_called()

    def test_compute_feature_creation_failure(self, feature_manager, sample_dataframe):
        """
        Given: Feature creation fails
        When: compute_feature is called
        Then: None is returned
        """
        # Given
        feature_manager.feature_factory.create_feature.return_value = None

        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash"

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert result is None

    def test_compute_feature_computation_exception(self, feature_manager, sample_dataframe):
        """
        Given: Feature computation raises exception
        When: compute_feature is called
        Then: None is returned and error is logged
        """
        # Given
        bad_feature = MagicMock()
        bad_feature.compute_all.side_effect = Exception("Computation failed")
        feature_manager.feature_factory.create_feature.return_value = bad_feature

        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True

        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash"

        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert result is None


class TestDelayedComputation:
    """Test cases for delayed computation operations."""

    def test_compute_feature_delayed(self, feature_manager, sample_dataframe):
        """
        Given: Valid feature parameters
        When: compute_feature_delayed is called
        Then: Delayed object is returned
        """
        # Given
        feature_def = MagicMock(spec=FeatureDefinition)
        param_set = MagicMock(spec=BaseParameterSetConfig)
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)

        # When
        result = feature_manager.compute_feature_delayed(feature_def, param_set, sample_dataframe, dataset_id)

        # Then
        assert isinstance(result, Delayed)

    def test_compute_latest_delayed(self, feature_manager):
        """
        Given: Feature instance
        When: compute_latest_delayed is called
        Then: Delayed object is returned
        """
        # Given
        feature = MockFeature("TestFeature")

        # When
        result = feature_manager.compute_latest_delayed(feature)
          # Then
        assert isinstance(result, Delayed)

    def test_compute_features_latest_delayed(self, feature_manager, sample_dataset_id, sample_dataset_id_2):
        """
        Given: Multiple features in manager
        When: compute_features_latest_delayed is called
        Then: List of delayed objects is returned
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")

        feature_key1 = create_feature_key("Feature1", sample_dataset_id, "hash1")
        feature_key2 = create_feature_key("Feature2", sample_dataset_id_2, "hash2")

        feature_manager._features[feature_key1] = feature1
        feature_manager._features[feature_key2] = feature2

        # When
        result = feature_manager.compute_features_latest_delayed()

        # Then
        assert len(result) == 2
        assert all(isinstance(delayed_task, Delayed) for delayed_task in result)

    def test_compute_features_latest_delayed_empty(self, feature_manager):
        """
        Given: No features in manager
        When: compute_features_latest_delayed is called
        Then: Empty list is returned
        """
        # When
        result = feature_manager.compute_features_latest_delayed()

        # Then
        assert result == []


class TestComputeAll:
    """Test cases for compute_all functionality."""

    def test_compute_all_success(self, feature_manager):
        """
        Given: Multiple features exist in manager
        When: compute_all is called
        Then: Combined DataFrame is returned
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")
        feature_manager._features[("Feature1", "hash1")] = feature1
        feature_manager._features[("Feature2", "hash2")] = feature2

        # When
        result = feature_manager.compute_all()

        # Then
        assert result is not None
        assert isinstance(result, DataFrame)

    def test_compute_all_no_features(self, feature_manager):
        """
        Given: No features exist in manager
        When: compute_all is called
        Then: None is returned and warning is logged
        """
        # When
        result = feature_manager.compute_all()

        # Then
        assert result is None

    @patch('drl_trading_core.core.service.feature_manager.compute')
    def test_compute_all_with_dask_exception(self, mock_compute, feature_manager):
        """
        Given: Dask compute raises exception
        When: compute_all is called
        Then: None is returned and error is logged
        """
        # Given
        feature = MockFeature("TestFeature")
        feature_manager._features[("TestFeature", "hash1")] = feature
        mock_compute.side_effect = Exception("Dask compute failed")

        # When
        result = feature_manager.compute_all()

        # Then
        assert result is None

    @patch('drl_trading_core.core.service.feature_manager.compute')
    def test_compute_all_filters_none_results(self, mock_compute, feature_manager):
        """
        Given: Some features return None results
        When: compute_all is called
        Then: None results are filtered out
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")
        feature_manager._features[("Feature1", "hash1")] = feature1
        feature_manager._features[("Feature2", "hash2")] = feature2

        # Mock compute to return one None and one valid DataFrame
        valid_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_compute.return_value = [None, valid_df]

        # When
        result = feature_manager.compute_all()

        # Then
        assert result is not None
        assert result.equals(valid_df)

    @patch('drl_trading_core.core.service.feature_manager.compute')
    def test_compute_all_no_valid_results(self, mock_compute, feature_manager):
        """
        Given: All features return None or empty results
        When: compute_all is called
        Then: None is returned and warning is logged
        """
        # Given
        feature = MockFeature("TestFeature")
        feature_manager._features[("TestFeature", "hash1")] = feature
        mock_compute.return_value = [None, pd.DataFrame()]  # None and empty DataFrame

        # When
        result = feature_manager.compute_all()

        # Then
        assert result is None


class TestComputeLatest:
    """Test cases for compute_latest functionality."""

    def test_compute_latest_success(self, feature_manager):
        """
        Given: Multiple features exist in manager
        When: compute_latest is called
        Then: Combined DataFrame with latest values is returned
        """
        # Given
        feature1 = MockFeature("Feature1")
        feature2 = MockFeature("Feature2")
        feature_manager._features[("Feature1", "hash1")] = feature1
        feature_manager._features[("Feature2", "hash2")] = feature2

        # When
        result = feature_manager.compute_latest()

        # Then
        assert result is not None
        assert isinstance(result, DataFrame)

    def test_compute_latest_no_features(self, feature_manager):
        """
        Given: No features exist in manager
        When: compute_latest is called
        Then: None is returned and warning is logged
        """
        # When
        result = feature_manager.compute_latest()

        # Then
        assert result is None

    @patch('drl_trading_core.core.service.feature_manager.compute')
    def test_compute_latest_with_dask_exception(self, mock_compute, feature_manager):
        """
        Given: Dask compute raises exception
        When: compute_latest is called
        Then: None is returned and error is logged
        """
        # Given
        feature = MockFeature("TestFeature")
        feature_manager._features[("TestFeature", "hash1")] = feature
        mock_compute.side_effect = Exception("Dask compute failed")

        # When
        result = feature_manager.compute_latest()

        # Then
        assert result is None


class TestDataFrameCombination:
    """Test cases for DataFrame combination utilities."""

    def test_combine_dataframes_efficiently_single_dataframe(self, feature_manager):
        """
        Given: Single DataFrame in list
        When: _combine_dataframes_efficiently is called
        Then: Single DataFrame is returned unchanged
        """
        # Given
        df = pd.DataFrame({"test": [1, 2, 3]})

        # When
        result = feature_manager._combine_dataframes_efficiently([df])

        # Then
        assert result is not None
        assert result.equals(df)

    def test_combine_dataframes_efficiently_multiple_dataframes(self, feature_manager):
        """
        Given: Multiple DataFrames in list
        When: _combine_dataframes_efficiently is called
        Then: Combined DataFrame is returned
        """
        # Given
        df1 = pd.DataFrame({"feature1": [1, 2, 3]})
        df2 = pd.DataFrame({"feature2": [4, 5, 6]})

        # When
        result = feature_manager._combine_dataframes_efficiently([df1, df2])

        # Then
        assert result is not None
        assert "feature1" in result.columns
        assert "feature2" in result.columns

    def test_combine_dataframes_efficiently_empty_list(self, feature_manager):
        """
        Given: Empty list of DataFrames
        When: _combine_dataframes_efficiently is called
        Then: None is returned
        """
        # When
        result = feature_manager._combine_dataframes_efficiently([])

        # Then
        assert result is None

    @patch('pandas.concat')
    def test_combine_dataframes_efficiently_concat_fallback(self, mock_concat, feature_manager):
        """
        Given: pandas.concat raises exception
        When: _combine_dataframes_efficiently is called
        Then: Fallback to iterative join is used
        """
        # Given
        df1 = pd.DataFrame({"feature1": [1, 2, 3]})
        df2 = pd.DataFrame({"feature2": [4, 5, 6]})
        mock_concat.side_effect = Exception("Concat failed")

        # When
        result = feature_manager._combine_dataframes_efficiently([df1, df2])

        # Then
        assert result is not None
        # Should fallback to join operation
        assert "feature1" in result.columns
        assert "feature2" in result.columns
