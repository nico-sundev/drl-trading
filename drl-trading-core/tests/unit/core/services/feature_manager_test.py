"""
Unit tests for the FeatureManager class.

Tests the public interface and behavior, focusing on the API contract
rather than internal implementation details.
"""
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from drl_trading_common.config.feature_config import FeatureDefinition, FeaturesConfig
from drl_trading_common.config.dask_config import DaskConfig
from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.decorator.feature_role_decorator import feature_role
from pandas import DataFrame

from drl_trading_core.core.service.feature_manager import FeatureManager, FeatureKey
from drl_trading_core.core.service.feature_definition_parser import FeatureDefinitionParser
from drl_trading_core.core.config.feature_computation_config import FeatureComputationConfig


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def __init__(self, feature_name: str = "MockFeature"):
        self.feature_name = feature_name
        # Create data with datetime index for is_caught_up testing
        self.data = pd.DataFrame({
            "value": [1.0, 2.0, 3.0]
        }, index=pd.date_range("2024-01-01 10:00:00", periods=3, freq="1min"))

        # Mock dataset_id for is_caught_up functionality
        from drl_trading_common.model.dataset_identifier import DatasetIdentifier
        from drl_trading_common.model.timeframe import Timeframe
        self.dataset_id = DatasetIdentifier("MOCK", Timeframe.MINUTE_1)

    def compute_all(self) -> DataFrame:
        return self.data.copy()

    def compute_latest(self) -> DataFrame:
        return self.data.tail(1).copy()

    def update(self, df: DataFrame) -> None:
        # Preserve the datetime index when updating
        if self.data is not None:
            self.data = pd.concat([self.data, df], ignore_index=False)
        else:
            self.data = df.copy()

    def get_feature_name(self) -> str:
        return self.feature_name

    def get_sub_features_names(self) -> list[str]:
        return [self.feature_name]

    def get_config_to_string(self) -> str:
        return f"MockFeature_{self.feature_name}"


class TestFeatureManager:
    """Unit tests for the FeatureManager public interface."""

    @pytest.fixture
    def mock_feature_factory(self):
        """Create a mock feature factory."""
        factory = MagicMock(spec=IFeatureFactory)

        # Configure factory to return mock features
        def create_feature_side_effect(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}_{dataset_id.timeframe.value}")
            # Set the dataset_id to match the actual one passed to the factory
            feature.dataset_id = dataset_id
            return feature

        factory.create_feature.side_effect = create_feature_side_effect
        return factory

    @pytest.fixture
    def mock_feature_definition_parser(self) -> MagicMock:
        """Create a mock feature definition parser."""
        return MagicMock(spec=FeatureDefinitionParser)

    @pytest.fixture
    def feature_computation_config(self) -> FeatureComputationConfig:
        """Create a feature computation config for testing."""
        return FeatureComputationConfig(
            dask=DaskConfig(
                scheduler="synchronous",  # Use synchronous for deterministic tests
                num_workers=1,
                memory_limit_per_worker_mb=512,
            )
        )

    @pytest.fixture
    def sample_features_config(self):
        """Create a sample features configuration."""
        # Create a mock parameter set
        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash_123"

        # Create a feature definition
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True
        feature_def.parameter_sets = [{"period": 14}]
        feature_def.parsed_parameter_sets = {"test_hash_123": param_set}

        # Create features config
        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def]
        config.dataset_definitions = {
            "BTCUSD": [Timeframe.MINUTE_1],
            "ETHUSD": [Timeframe.MINUTE_5]
        }

        return config

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample market data."""
        return pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200],
            "timestamp": pd.date_range("2023-01-01", periods=3)
        })

    def test_feature_manager_initialization(
        self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config
    ):
        """Test FeatureManager can be initialized with all required dependencies."""
        # Given & When
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # Then
        assert manager.feature_factory == mock_feature_factory
        assert manager.feature_computation_config == feature_computation_config
        assert len(manager._features) == 0

    def test_feature_manager_update_creates_features(
        self, mock_feature_factory, mock_feature_definition_parser,
        feature_computation_config, sample_features_config, sample_dataframe
    ):
        """Test that calling request_features_update creates and stores features correctly."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, sample_features_config)

        # Then
        assert len(manager._features) > 0
        # Should create features for each dataset (BTCUSD, ETHUSD)
        assert mock_feature_factory.create_feature.call_count >= 2

    def test_feature_manager_compute_all(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test that compute_all returns combined feature data."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When
        result = manager.compute_all()

        # Then
        assert result is not None
        assert isinstance(result, DataFrame)
        assert not result.empty

    def test_feature_manager_compute_latest(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test that compute_latest returns latest feature values."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When
        result = manager.compute_latest()

        # Then
        assert result is not None
        assert isinstance(result, DataFrame)
        assert not result.empty

    def test_feature_key_functionality(self):
        """Test FeatureKey creation and string representation."""
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash"
        )

        # When & Then
        assert feature_key.feature_name == "TestFeature"
        assert feature_key.dataset_id.symbol == "BTCUSD"
        assert feature_key.dataset_id.timeframe == Timeframe.MINUTE_1
        assert feature_key.param_hash == "test_hash"

        # Test string representation
        key_string = feature_key.to_string()
        assert "TestFeature_BTCUSD_1m_test_hash" == key_string

    def test_compute_all_no_features_returns_none(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config):
        """Test that compute_all returns None when no features are initialized."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        result = manager.compute_all()

        # Then
        assert result is None

    def test_compute_latest_no_features_returns_none(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config):
        """Test that compute_latest returns None when no features are initialized."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        result = manager.compute_latest()

        # Then
        assert result is None

    def test_update_multiple_times_accumulates_data(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test that calling request_features_update multiple times properly accumulates data in features."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, sample_features_config)
        manager.update(sample_dataframe)  # Update again with just data (features already initialized)

        # Then
        assert len(manager._features) > 0
        # Verify that update was called on existing features
        result = manager.compute_all()
        assert result is not None

    def test_feature_manager_handles_disabled_features(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_dataframe):
        """Test that disabled features are not created."""
        # Given
        param_set = MagicMock(spec=BaseParameterSetConfig)
        param_set.enabled = True
        param_set.hash_id.return_value = "test_hash_123"

        # Create a disabled feature definition
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "DisabledFeature"
        feature_def.enabled = False  # Disabled
        feature_def.parameter_sets = [{"period": 14}]
        feature_def.parsed_parameter_sets = {"test_hash_123": param_set}

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def]
        config.dataset_definitions = {"BTCUSD": [Timeframe.MINUTE_1]}

        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, config)

        # Then
        assert len(manager._features) == 0  # No features should be created

    def test_update_only_updates_existing_features(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test that update() method only updates data in existing features."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)  # Initialize features
        initial_feature_count = len(manager._features)

        # When
        manager.update(sample_dataframe)  # Only update data, no new features

        # Then
        assert len(manager._features) == initial_feature_count  # Same number of features
        # Verify compute still works (features were updated with new data)
        result = manager.compute_all()
        assert result is not None

    def test_feature_key_repr_method(self):
        """Test FeatureKey __repr__ method for debugging."""
        # Given
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash"
        )

        # When
        repr_str = repr(feature_key)

        # Then
        assert "FeatureKey(feature='TestFeature'" in repr_str
        assert "symbol='BTCUSD'" in repr_str
        assert "timeframe='1m'" in repr_str
        assert "param_hash='test_hash'" in repr_str

    def test_feature_creation_with_factory_exception(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test handling of factory exceptions during feature creation."""
        # Given
        mock_feature_factory.create_feature.side_effect = Exception("Factory error")
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, sample_features_config)

        # Then
        assert len(manager._features) == 0  # No features created due to exception
        assert manager._feature_creation_stats["creation_failures"] > 0

    def test_compute_all_with_empty_dataframe_results(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test compute_all when features return empty DataFrames."""
        # Given
        def create_feature_with_empty_data(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            feature.data = pd.DataFrame()  # Empty DataFrame
            return feature

        mock_feature_factory.create_feature.side_effect = create_feature_with_empty_data
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When
        result = manager.compute_all()

        # Then
        # The system actually combines empty DataFrames, so result exists but may be empty
        assert result is not None
        # Check that it handled empty DataFrames without crashing
        assert isinstance(result, pd.DataFrame)

    def test_feature_storage_with_duplicate_keys(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test handling of duplicate feature keys during storage."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # Create a feature manually to simulate existing feature
        existing_feature = MockFeature("ExistingFeature")
        dataset_id = DatasetIdentifier("BTCUSD", Timeframe.MINUTE_1)
        feature_key = FeatureKey(
            feature_name="TestFeature",
            dataset_id=dataset_id,
            param_hash="test_hash_123"
        )
        manager._features[feature_key] = existing_feature

        # When - try to create features that would conflict
        manager.request_features_update(sample_dataframe, sample_features_config)

        # Then - should handle conflicts gracefully (overwrites)
        assert len(manager._features) > 0

    def test_feature_creation_returns_none(self, mock_feature_definition_parser, feature_computation_config, sample_dataframe):
        """Test handling when factory returns None for feature creation."""
        # Given
        mock_factory = MagicMock(spec=IFeatureFactory)
        mock_factory.create_feature.return_value = None

        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True
        feature_def.parameter_sets = []
        feature_def.parsed_parameter_sets = {}

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def]
        config.dataset_definitions = {"BTCUSD": [Timeframe.MINUTE_1]}

        manager = FeatureManager(
            feature_factory=mock_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, config)

        # Then
        assert len(manager._features) == 0
        assert manager._feature_creation_stats["creation_failures"] > 0
        assert manager._feature_creation_stats["successfully_created"] == 0

    def test_update_features_data_with_exception(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test update_features_data when feature.update() raises exception."""
        # Given
        def create_bad_feature(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Override update method to raise exception
            def bad_update(df):
                raise Exception("Update failed")
            feature.update = bad_update
            return feature

        mock_feature_factory.create_feature.side_effect = create_bad_feature
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When - update with new data should handle exceptions
        new_data = pd.DataFrame({"close": [110.0, 111.0]})
        manager.update(new_data)  # Should not crash despite exceptions

        # Then - manager should still be operational
        assert len(manager._features) > 0

    def test_compute_latest_with_feature_exceptions(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test compute_latest when features raise exceptions during computation."""
        # Given
        def create_feature_with_bad_compute_latest(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Override compute_latest to raise exception
            def bad_compute_latest():
                raise Exception("Compute latest failed")
            feature.compute_latest = bad_compute_latest
            return feature

        mock_feature_factory.create_feature.side_effect = create_feature_with_bad_compute_latest
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When
        result = manager.compute_latest()

        # Then - should handle exceptions gracefully
        assert result is None  # No valid results due to exceptions

    def test_combine_dataframes_with_different_lengths(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test DataFrame combination with features returning different length DataFrames."""
        # Given
        call_count = [0]
        def create_feature_with_varying_data(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            call_count[0] += 1
            # Create DataFrames with different lengths
            if call_count[0] % 2 == 0:
                feature.data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})  # 4 rows
            else:
                feature.data = pd.DataFrame({"value": [1.0, 2.0]})  # 2 rows
            return feature

        mock_feature_factory.create_feature.side_effect = create_feature_with_varying_data
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # When
        result = manager.compute_all()

        # Then - should handle different lengths gracefully
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_features_with_no_parameter_sets(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_dataframe):
        """Test features that have no parameter sets (simple features)."""
        # Given
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "SimpleFeature"
        feature_def.enabled = True
        feature_def.parameter_sets = []  # No parameter sets
        feature_def.parsed_parameter_sets = {}  # Empty dict

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def]
        config.dataset_definitions = {"BTCUSD": [Timeframe.MINUTE_1]}

        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )

        # When
        manager.request_features_update(sample_dataframe, config)

        # Then
        assert len(manager._features) > 0  # Should create feature with no_config hash
        # Check that the feature was created with NO_CONFIG_HASH
        feature_keys = list(manager._features.keys())
        assert any(key.param_hash == "no_config" for key in feature_keys)

    def test_dask_compute_exception_handling(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test handling of Dask computation exceptions."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # Mock dask.compute to raise an exception
        with patch('drl_trading_core.core.service.feature_manager.compute') as mock_compute:
            mock_compute.side_effect = Exception("Dask computation failed")

            # When
            result = manager.compute_all()

            # Then
            assert result is None  # Should return None when computation fails

    def test_is_caught_up_no_features(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config):
        """Test is_caught_up when no features are initialized."""
        # Given
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = manager.are_features_caught_up(reference_time)

        # Then
        assert result is False

    def test_is_caught_up_all_features_caught_up(self, mock_feature_definition_parser, feature_computation_config):
        """Test is_caught_up when all features are caught up."""
        # Given
        # Create a factory that produces features with proper datetime indices
        factory = MagicMock(spec=IFeatureFactory)

        def create_caught_up_feature(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            feature.dataset_id = dataset_id
            # Ensure data has proper datetime index with recent timestamps
            feature.data = pd.DataFrame({
                "value": [1.0, 2.0, 3.0]
            }, index=pd.date_range("2024-01-01 10:00:00", periods=3, freq="1min"))
            return feature

        factory.create_feature.side_effect = create_caught_up_feature

        # Create simple config
        feature_def = MagicMock(spec=FeatureDefinition)
        feature_def.name = "TestFeature"
        feature_def.enabled = True
        feature_def.parameter_sets = []
        feature_def.parsed_parameter_sets = {}

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def]
        config.dataset_definitions = {"BTCUSD": [Timeframe.MINUTE_1]}

        sample_df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [104.0], "volume": [1000]
        }, index=pd.date_range("2024-01-01 10:03:00", periods=1, freq="1min"))

        manager = FeatureManager(
            feature_factory=factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_df, config)

        # Reference time is within 1 minute of last data point (10:02:00)
        reference_time = datetime(2024, 1, 1, 10, 2, 30)

        # When
        result = manager.are_features_caught_up(reference_time)

        # Then
        assert result is True

    def test_is_caught_up_some_features_not_caught_up(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test is_caught_up when some features are not caught up."""
        # Given
        def create_mixed_caught_up_features(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Simulate some features being older than others
            if "BTCUSD" in feature.feature_name:
                # This feature has old data - not caught up
                feature.data = pd.DataFrame({
                    "value": [1.0, 2.0]
                }, index=pd.date_range("2024-01-01 09:00:00", periods=2, freq="1min"))
            return feature

        mock_feature_factory.create_feature.side_effect = create_mixed_caught_up_features
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        # Reference time is much later - some features won't be caught up
        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = manager.are_features_caught_up(reference_time)

        # Then
        assert result is False

    def test_is_caught_up_feature_exception_handling(self, mock_feature_factory, mock_feature_definition_parser, feature_computation_config, sample_features_config, sample_dataframe):
        """Test is_caught_up when feature.is_caught_up raises exception."""
        # Given
        def create_feature_with_bad_is_caught_up(feature_name, dataset_id, config=None, postfix=""):
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Override is_caught_up to raise exception
            def bad_is_caught_up(reference_time):
                raise Exception("is_caught_up failed")
            feature.are_features_caught_up = bad_is_caught_up
            return feature

        mock_feature_factory.create_feature.side_effect = create_feature_with_bad_is_caught_up
        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        manager.request_features_update(sample_dataframe, sample_features_config)

        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = manager.are_features_caught_up(reference_time)

        # Then
        assert result is False  # Should return False when exceptions occur

    def test_base_feature_is_caught_up_within_timeframe(self):
        """Test BaseFeature.is_caught_up when feature is caught up."""
        # Given
        feature = MockFeature("TestFeature")
        # Last data point is at 10:02:00, reference time is 10:02:30 (30 seconds later)
        reference_time = datetime(2024, 1, 1, 10, 2, 30)

        # When
        result = feature.are_features_caught_up(reference_time)

        # Then
        assert result is True  # 30 seconds < 60 seconds (1 minute timeframe)

    def test_base_feature_is_caught_up_outside_timeframe(self):
        """Test BaseFeature.is_caught_up when feature is not caught up."""
        # Given
        feature = MockFeature("TestFeature")
        # Last data point is at 10:02:00, reference time is 10:05:00 (3 minutes later)
        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = feature.are_features_caught_up(reference_time)

        # Then
        assert result is False  # 3 minutes > 1 minute timeframe

    def test_base_feature_is_caught_up_no_data(self):
        """Test BaseFeature.is_caught_up when feature has no data."""
        # Given
        feature = MockFeature("TestFeature")
        feature.data = None
        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = feature.are_features_caught_up(reference_time)

        # Then
        assert result is False

    def test_base_feature_is_caught_up_empty_data(self):
        """Test BaseFeature.is_caught_up when feature has empty data."""
        # Given
        feature = MockFeature("TestFeature")
        feature.data = pd.DataFrame()  # Empty DataFrame
        reference_time = datetime(2024, 1, 1, 10, 5, 0)

        # When
        result = feature.are_features_caught_up(reference_time)

        # Then
        assert result is False


class TestFeatureManagerEventTimestampDeduplication:
    """Test cases for event_timestamp deduplication when combining multiple features."""

    @pytest.fixture
    def mock_feature_factory(self) -> MagicMock:
        """Create a mock feature factory."""
        return MagicMock(spec=IFeatureFactory)

    @pytest.fixture
    def mock_feature_definition_parser(self) -> MagicMock:
        """Create a mock feature definition parser."""
        return MagicMock(spec=FeatureDefinitionParser)

    @pytest.fixture
    def feature_computation_config(self) -> FeatureComputationConfig:
        """Create a feature computation config for testing."""
        return FeatureComputationConfig(
            dask=DaskConfig(
                scheduler="synchronous",
                num_workers=1,
                memory_limit_per_worker_mb=512,
            )
        )

    def test_combine_dataframes_deduplicates_event_timestamp(
        self,
        mock_feature_factory: MagicMock,
        mock_feature_definition_parser: MagicMock,
        feature_computation_config: FeatureComputationConfig,
    ) -> None:
        """Test that _combine_dataframes_efficiently removes duplicate event_timestamp columns.

        This is a regression test for the issue where each feature's compute_all() returns
        a DataFrame with an event_timestamp column, causing pd.concat to create duplicates.
        """
        # Given: Multiple features that each return DataFrames with event_timestamp
        timestamps = pd.date_range("2024-01-01 10:00:00", periods=3, freq="1h")

        def create_feature_with_event_timestamp(feature_name: str, dataset_id, config=None, postfix=""):
            """Create a feature that returns DataFrame with event_timestamp (realistic scenario)."""
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Simulate realistic feature output: event_timestamp + feature columns
            feature.data = pd.DataFrame({
                "event_timestamp": timestamps,
                f"{feature_name}_value": [10.0 + i for i in range(3)]
            })
            return feature

        mock_feature_factory.create_feature.side_effect = create_feature_with_event_timestamp

        # Create config with multiple features
        feature_def_1 = MagicMock(spec=FeatureDefinition)
        feature_def_1.name = "rsi"
        feature_def_1.enabled = True
        feature_def_1.parameter_sets = []
        feature_def_1.parsed_parameter_sets = {}

        feature_def_2 = MagicMock(spec=FeatureDefinition)
        feature_def_2.name = "sma"
        feature_def_2.enabled = True
        feature_def_2.parameter_sets = []
        feature_def_2.parsed_parameter_sets = {}

        feature_def_3 = MagicMock(spec=FeatureDefinition)
        feature_def_3.name = "bb_upper"
        feature_def_3.enabled = True
        feature_def_3.parameter_sets = []
        feature_def_3.parsed_parameter_sets = {}

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def_1, feature_def_2, feature_def_3]
        config.dataset_definitions = {"EURUSD": [Timeframe.MINUTE_5]}

        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        sample_data = pd.DataFrame({"close": [1.1, 1.2, 1.3]})

        # When: Request features update and compute all
        manager.request_features_update(sample_data, config)
        result = manager.compute_all()

        # Then: Should have exactly ONE event_timestamp column
        assert result is not None, "Result should not be None"
        assert "event_timestamp" in result.columns, "Result must contain event_timestamp"

        event_timestamp_count = list(result.columns).count("event_timestamp")
        assert event_timestamp_count == 1, (
            f"Expected exactly 1 event_timestamp column, found {event_timestamp_count}. "
            f"Columns: {list(result.columns)}"
        )

        # Then: Should have all feature columns
        assert "rsi_value" in result.columns
        assert "sma_value" in result.columns
        assert "bb_upper_value" in result.columns

        # Then: Verify total column count (1 event_timestamp + 3 feature columns + 1 close from sample_data)
        assert len(result.columns) == 5, f"Expected 5 columns, got {len(result.columns)}: {list(result.columns)}"

    def test_combine_dataframes_handles_mixed_columns(
        self,
        mock_feature_factory: MagicMock,
        mock_feature_definition_parser: MagicMock,
        feature_computation_config: FeatureComputationConfig,
    ) -> None:
        """Test deduplication with mixed columns including symbol and event_timestamp."""
        # Given: Features with common columns (event_timestamp, symbol) + unique feature columns
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1h")

        def create_realistic_feature(feature_name: str, dataset_id, config=None, postfix=""):
            """Create feature with realistic Feast-compatible structure."""
            feature = MockFeature(f"{feature_name}_{dataset_id.symbol}")
            # Realistic: event_timestamp + symbol + feature columns
            feature.data = pd.DataFrame({
                "event_timestamp": timestamps,
                "symbol": [dataset_id.symbol] * len(timestamps),
                f"{feature_name}_col1": [float(i) for i in range(len(timestamps))],
                f"{feature_name}_col2": [float(i * 2) for i in range(len(timestamps))]
            })
            return feature

        mock_feature_factory.create_feature.side_effect = create_realistic_feature

        # Create config with two features
        feature_def_1 = MagicMock(spec=FeatureDefinition)
        feature_def_1.name = "rsi"
        feature_def_1.enabled = True
        feature_def_1.parameter_sets = []
        feature_def_1.parsed_parameter_sets = {}

        feature_def_2 = MagicMock(spec=FeatureDefinition)
        feature_def_2.name = "macd"
        feature_def_2.enabled = True
        feature_def_2.parameter_sets = []
        feature_def_2.parsed_parameter_sets = {}

        config = MagicMock(spec=FeaturesConfig)
        config.feature_definitions = [feature_def_1, feature_def_2]
        config.dataset_definitions = {"EURUSD": [Timeframe.HOUR_1]}

        manager = FeatureManager(
            feature_factory=mock_feature_factory,
            feature_definition_parser=mock_feature_definition_parser,
            feature_computation_config=feature_computation_config,
        )
        sample_data = pd.DataFrame({"close": [1.1, 1.2, 1.3, 1.4, 1.5]})

        # When: Compute features
        manager.request_features_update(sample_data, config)
        result = manager.compute_all()

        # Then: Should deduplicate common columns
        assert result is not None
        assert list(result.columns).count("event_timestamp") == 1, "Should have exactly 1 event_timestamp"
        assert list(result.columns).count("symbol") == 1, "Should have exactly 1 symbol"

        # Then: Should have all feature-specific columns
        assert "rsi_col1" in result.columns
        assert "rsi_col2" in result.columns
        assert "macd_col1" in result.columns
        assert "macd_col2" in result.columns

        # Then: Total columns = 2 common (event_timestamp, symbol) + 4 feature columns + 1 close from sample_data = 7
        assert len(result.columns) == 7, f"Expected 7 columns, got {len(result.columns)}: {list(result.columns)}"
