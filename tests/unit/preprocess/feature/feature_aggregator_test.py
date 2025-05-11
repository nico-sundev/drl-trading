"""Unit tests for the FeatureAggregator class."""

from unittest.mock import MagicMock, patch

import dask
import pandas as pd
import pytest
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config import FeatureDefinition, FeaturesConfig
from ai_trading.config.feature_config_factory import FeatureConfigFactoryInterface
from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import (
    FeastService,
    FeastServiceInterface,
)
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
    FeatureAggregatorInterface,
)
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry


class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def compute(self) -> DataFrame:
        """Generate mock feature data."""
        # Make a copy preserving the index
        df = self.df_source.copy()
        df["feature1"] = 1.0
        df["feature2"] = 2.0
        if self.config.name == "drop_time":
            # Return a DataFrame without index name
            df.index.name = None
            return df
        else:
            return df

    def get_sub_features_names(self) -> list[str]:
        """Return mock sub-feature names."""
        return ["feature1", "feature2"]

    def get_feature_name(self) -> str:
        """Return the base name of the feature."""
        return "MockFeature"


@pytest.fixture
def mock_param_set() -> BaseParameterSetConfig:
    """Create a mock parameter set for features."""
    param_set = MagicMock(spec=BaseParameterSetConfig)
    param_set.enabled = True
    param_set.hash_id.return_value = "abc123hash"
    param_set.to_string.return_value = "7_14"
    param_set.name = "default_params"
    return param_set


@pytest.fixture
def mock_param_set_drop_time() -> BaseParameterSetConfig:
    """Create a mock parameter set that causes MockFeature to drop Time."""
    param_set = MagicMock(spec=BaseParameterSetConfig)
    param_set.enabled = True
    param_set.hash_id.return_value = "droptimehash"
    param_set.name = "drop_time"
    return param_set


@pytest.fixture
def mock_feature_definition(mock_param_set) -> FeatureDefinition:
    """Create a mock feature definition."""
    # Mock the FeatureConfigFactory before creating the FeatureDefinition
    with patch(
        "ai_trading.config.feature_config_factory.FeatureConfigFactory"
    ) as mock_factory_class:
        # Create mock instance with get_config_class method
        mock_factory_instance = MagicMock(spec=FeatureConfigFactoryInterface)
        mock_factory_class.return_value = mock_factory_instance

        # Mock config class for 'mockfeature'
        class MockFeatureConfig(BaseParameterSetConfig):
            pass

        # Set up the mock factory to return our mock config class
        mock_factory_instance.get_config_class.return_value = MockFeatureConfig

        # Now create the FeatureDefinition which will use our mocked factory
        mock_feature_def = FeatureDefinition(
            name="MockFeature",
            enabled=True,
            derivatives=[],
            parameter_sets=[],  # Empty raw params
        )

        # Manually set the parsed_parameter_sets since we're bypassing the validator
        mock_feature_def.parsed_parameter_sets = [mock_param_set]
        return mock_feature_def


@pytest.fixture
def mock_features_config(mock_feature_definition: FeatureDefinition) -> FeaturesConfig:
    """Create a mock features configuration."""
    return FeaturesConfig(feature_definitions=[mock_feature_definition])


@pytest.fixture
def mock_asset_df() -> DataFrame:
    """Create a mock DataFrame for asset price data."""
    dates = pd.date_range(start="2022-01-01", periods=10, freq="H")
    data = {
        "Open": [1.0] * 10,
        "High": [2.0] * 10,
        "Low": [0.5] * 10,
        "Close": [1.5] * 10,
        "Volume": [1000.0] * 10,
    }
    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def mock_asset_data(mock_asset_df) -> AssetPriceDataSet:
    """Create a mock asset price dataset."""
    asset_data = AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=mock_asset_df,
    )
    # Add symbol as an attribute after initialization
    return asset_data


@pytest.fixture
def mock_symbol() -> str:
    """Create a mock symbol for testing."""
    return "EURUSD"


@pytest.fixture
def mock_class_registry() -> FeatureClassRegistry:
    """Create a mock feature class registry."""
    registry = MagicMock(spec=FeatureClassRegistry)
    registry.feature_class_map = {"MockFeature": MockFeature}
    return registry


@pytest.fixture
def mock_feast_service() -> FeastServiceInterface:
    """Create a mock FeastService that implements FeastServiceInterface."""
    feast_service = MagicMock(spec=FeastService)
    feast_service.is_enabled.return_value = True
    feast_service.get_historical_features.return_value = None
    return feast_service


@pytest.fixture
def feature_aggregator(
    mock_features_config, mock_class_registry, mock_feast_service
) -> FeatureAggregator:
    """Create a FeatureAggregator instance with mocked dependencies."""
    return FeatureAggregator(
        config=mock_features_config,
        class_registry=mock_class_registry,
        feast_service=mock_feast_service,
    )


def test_compute_single_feature_no_cache(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set,
    mock_feast_service,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature when feature is not cached."""
    # Given
    # Feature not in cache
    mock_feast_service.get_historical_features.return_value = None

    # When
    # Compute feature without cache
    result_df = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify feature was retrieved from Feast and stored back
    mock_feast_service.get_historical_features.assert_called_once_with(
        feature_name="MockFeature",
        param_hash="abc123hash",
        sub_feature_names=["feature1", "feature2"],
        asset_data=mock_asset_data,
        symbol=mock_symbol,
    )
    mock_feast_service.store_computed_features.assert_called_once()
    assert result_df is not None
    assert not result_df.empty
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert result_df.index.name == "Time"
    expected_col1 = "feature1"
    expected_col2 = "feature2"
    assert expected_col1 in result_df.columns
    assert expected_col2 in result_df.columns
    assert "Open" not in result_df.columns
    assert len(result_df.columns) == 2
    pd.testing.assert_index_equal(
        result_df.index,
        mock_asset_df.index,
    )


def test_compute_single_feature_with_cache(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set,
    mock_feast_service,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature when feature is cached."""
    # Given
    # Feature exists in cache with DatetimeIndex
    cached_features = DataFrame(
        {
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
        },
        index=mock_asset_df.index.copy(),
    )
    cached_features.index.name = "Time"
    mock_feast_service.get_historical_features.return_value = cached_features

    # When
    # Retrieve feature from cache
    result_df = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify feature was retrieved from cache and not computed again
    mock_feast_service.get_historical_features.assert_called_once()
    mock_feast_service.store_computed_features.assert_not_called()
    assert result_df is not None

    assert not result_df.empty
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert result_df.index.name == "Time"
    expected_col1 = "feature1"
    expected_col2 = "feature2"
    assert expected_col1 in result_df.columns
    assert expected_col2 in result_df.columns
    assert len(result_df.columns) == 2
    pd.testing.assert_index_equal(result_df.index, mock_asset_df.index)


def test_compute_single_feature_disabled_feature_def(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature returns None if feature def is disabled."""
    # Given
    # Feature definition is disabled
    mock_feature_definition.enabled = False

    # When
    # Attempt to compute the feature
    result = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify None is returned
    assert result is None


def test_compute_single_feature_disabled_param_set(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature returns None if param set is disabled."""
    # Given
    # Parameter set is disabled
    mock_param_set.enabled = False

    # When
    # Attempt to compute the feature
    result = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify None is returned
    assert result is None


def test_compute_single_feature_handles_computation_error(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set,
    mock_class_registry,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature returns None if computation fails."""
    # Given
    # Feature computation will raise an error
    mock_feature_instance = MagicMock()
    mock_feature_instance.compute.side_effect = ValueError("Computation failed")
    mock_feature_instance.get_feature_name.return_value = "MockFeature"
    mock_feature_instance.get_sub_features_names.return_value = ["f1"]

    mock_feature_class = MagicMock(return_value=mock_feature_instance)
    feature_aggregator.class_registry.feature_class_map["MockFeature"] = (
        mock_feature_class
    )

    # When
    # Attempt to compute the feature
    result = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify None is returned and compute was called
    assert result is None
    mock_feature_instance.compute.assert_called_once()


def test_compute_single_feature_handles_missing_time_index_after_compute(
    feature_aggregator,
    mock_feature_definition,
    mock_param_set_drop_time,
    mock_feast_service,
    mock_asset_df,
    mock_asset_data,
    mock_symbol,
):
    """Test _compute_or_get_single_feature returns None if 'Time' index name is missing after compute."""
    # Given
    # Feature not in cache and will drop Time index name when computed
    mock_feast_service.get_historical_features.return_value = None

    # When
    # Attempt to compute the feature
    result = feature_aggregator._compute_or_get_single_feature(
        mock_feature_definition,
        mock_param_set_drop_time,
        mock_asset_df,
        mock_symbol,
        mock_asset_data,
    )

    # Then
    # Verify None is returned because the index name was removed
    assert result is None
    mock_feast_service.get_historical_features.assert_called_once()


@patch(
    "ai_trading.preprocess.feature.feature_aggregator.delayed",
    side_effect=lambda fn: fn,
)
def test_compute_returns_callable_tasks(
    mock_delayed_patch, feature_aggregator, mock_asset_data, mock_symbol
):
    """Test that compute returns a list of callable tasks wrapping the correct method."""
    # Given
    # Feature aggregator with default configuration

    # When
    # Call compute() method with asset data and symbol
    tasks = feature_aggregator.compute(asset_data=mock_asset_data, symbol=mock_symbol)

    # Then
    # Verify tasks are created correctly
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    mock_delayed_patch.assert_called_once()
    # Check that the delayed function is _compute_or_get_single_feature (check function name)
    assert (
        mock_delayed_patch.call_args[0][0].__name__ == "_compute_or_get_single_feature"
    )


@patch(
    "ai_trading.preprocess.feature.feature_aggregator.delayed",
    side_effect=lambda fn: fn,
)
def test_compute_handles_multiple_param_sets(
    mock_delayed_patch, feature_aggregator, mock_asset_data, mock_symbol
):
    """Test compute creates tasks for multiple parameter sets."""
    # Given
    # Feature with multiple parameter sets
    mock_param_set2 = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set2.enabled = True
    mock_param_set2.hash_id.return_value = "def456hash"
    mock_param_set2.name = "other_params"
    feature_aggregator.config.feature_definitions[0].parsed_parameter_sets.append(
        mock_param_set2
    )

    # When
    # Call compute() method with asset data and symbol
    tasks = feature_aggregator.compute(asset_data=mock_asset_data, symbol=mock_symbol)

    # Then
    # Verify correct number of tasks are created
    assert isinstance(tasks, list)
    assert len(tasks) == 2
    assert mock_delayed_patch.call_count == 2


@patch(
    "ai_trading.preprocess.feature.feature_aggregator.delayed",
    side_effect=lambda fn: fn,
)
def test_compute_skips_disabled_features_and_params(
    mock_delayed_patch, feature_aggregator, mock_asset_data, mock_symbol
):
    """Test compute skips disabled features and parameter sets."""
    # Given
    # Setup with disabled features and parameter sets
    mock_param_set2 = MagicMock(spec=BaseParameterSetConfig)
    mock_param_set2.enabled = True
    mock_param_set2.hash_id.return_value = "p2"
    mock_param_set2.name = "p2"

    # Create second mock feature definition without relying on constructor validation
    with patch(
        "ai_trading.config.feature_config_factory.FeatureConfigFactory"
    ) as mock_factory_class:
        mock_factory_instance = MagicMock(spec=FeatureConfigFactoryInterface)
        mock_factory_class.return_value = mock_factory_instance

        class MockFeatureConfig(BaseParameterSetConfig):
            pass

        # Set up the mock factory to return our mock config class
        mock_factory_instance.get_config_class.return_value = MockFeatureConfig

        mock_feature_def2 = MagicMock(spec=FeatureDefinition)
        mock_feature_def2.name = "MockFeature2"
        mock_feature_def2.enabled = False
        mock_feature_def2.parsed_parameter_sets = [mock_param_set2]

        # Add the mock feature definition to the config
        feature_aggregator.config.feature_definitions.append(mock_feature_def2)
        feature_aggregator.class_registry.feature_class_map["MockFeature2"] = (
            MockFeature
        )

    # Disable the first feature's parameter set
    feature_aggregator.config.feature_definitions[0].parsed_parameter_sets[
        0
    ].enabled = False

    # When
    # Call compute() method with asset data and symbol
    tasks = feature_aggregator.compute(asset_data=mock_asset_data, symbol=mock_symbol)

    # Then
    # Verify no tasks are created for disabled features/parameters
    assert isinstance(tasks, list)
    assert len(tasks) == 0
    assert mock_delayed_patch.call_count == 0


@patch("ai_trading.preprocess.feature.feature_aggregator.delayed")
def test_compute_execute_tasks_and_check_column_names(
    mock_delayed,
    feature_aggregator: FeatureAggregatorInterface,
    mock_feast_service,
    mock_asset_data,
    mock_symbol,
    mock_asset_df,
):
    """Test executing the delayed tasks from compute and checking column names using real dask compute."""
    # Given
    # Setup feature aggregator and ensure feature is not in cache
    mock_feast_service.get_historical_features.return_value = None
    # Note: mock hash_id as a method, not a property
    mock_hash = "abc123hash"
    mock_to_string = "7_14"
    feature_aggregator.config.feature_definitions[0].parsed_parameter_sets[
        0
    ].hash_id = lambda: mock_hash
    feature_aggregator.config.feature_definitions[0].parsed_parameter_sets[
        0
    ].to_string = lambda: mock_to_string

    # Mock delayed to return a proper task that dask can compute
    mock_delayed.side_effect = lambda fn: lambda *args, **kwargs: fn(*args, **kwargs)

    # When
    # Get tasks from compute method
    tasks = feature_aggregator.compute(asset_data=mock_asset_data, symbol=mock_symbol)

    # Use real dask compute to execute the tasks
    computed_results = dask.compute(*tasks)

    # Filter out None results
    computed_dfs = [df for df in computed_results if df is not None]

    # Then
    # Verify we got results
    assert len(computed_dfs) > 0
    result_df = computed_dfs[0]

    # Verify column structure
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert len(result_df.columns) == 2  # 2 feature columns

    # Check specific feature column naming pattern
    expected_col1 = "feature1"
    expected_col2 = "feature2"

    # Verify column names follow the expected pattern
    assert expected_col1 in result_df.columns
    assert expected_col2 in result_df.columns

    # Verify data was passed to and from the store correctly
    mock_feast_service.get_historical_features.assert_called_once()
    mock_feast_service.store_computed_features.assert_called_once()
