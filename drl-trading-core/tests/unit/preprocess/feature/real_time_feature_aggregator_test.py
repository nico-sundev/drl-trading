"""Unit tests for the RealTimeFeatureAggregator class."""

from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.config.feature_config import FeatureDefinition, FeaturesConfig
from pandas import DataFrame, Series

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_core.preprocess.feast.feast_service import (
    FeastService,
    FeastServiceInterface,
)
from drl_trading_core.preprocess.feature.feature_factory import (
    FeatureFactoryInterface,
)
from drl_trading_core.preprocess.feature.real_time_feature_aggregator import (
    RealTimeFeatureAggregator,
    RealTimeFeatureAggregatorInterface,
)


class MockRealTimeFeature(BaseFeature):
    """Mock feature class for real-time testing."""

    def compute(self) -> DataFrame:
        """Generate mock feature data for real-time computation."""
        # Create features based on the last row of input data
        df = self.df_source.copy()

        # Simple moving average as mock feature
        df["rt_feature_1"] = df["Close"].rolling(window=2, min_periods=1).mean()
        df["rt_feature_2"] = df["Close"].rolling(window=3, min_periods=1).std()

        return df[["rt_feature_1", "rt_feature_2"]]

    def get_sub_features_names(self) -> List[str]:
        """Return mock sub-feature names."""
        return ["rt_feature_1", "rt_feature_2"]

    def get_feature_name(self) -> str:
        """Return the base name of the feature."""
        return "MockRealTimeFeature"


class MockParameterSetConfig(BaseParameterSetConfig):
    """Mock parameter set config for testing."""

    name: str = "default"
    length: int = 10

    def __init__(self, enabled=True, name="default", length=10):
        super().__init__(type="mock_config", enabled=enabled)
        self.name = name
        self.length = length
        self._param_string = f"{length}"

    @property
    def param_string(self) -> str:
        return self._param_string

    def hash_id(self) -> str:
        return f"hash_{self.name}_{self.length}"


@pytest.fixture
def mock_param_set() -> BaseParameterSetConfig:
    """Create a mock parameter set for features."""
    return MockParameterSetConfig(enabled=True, name="default", length=10)


@pytest.fixture
def mock_disabled_param_set() -> BaseParameterSetConfig:
    """Create a disabled mock parameter set."""
    return MockParameterSetConfig(enabled=False, name="disabled", length=5)


@pytest.fixture
def mock_feature_definition(mock_param_set) -> FeatureDefinition:
    """Create a mock feature definition for real-time testing."""

    mock_feature_def = FeatureDefinition(
        name="MockRealTimeFeature",
        enabled=True,
        derivatives=[],
        parameter_sets=[],
    )

    # Set parsed parameter sets manually
    mock_feature_def.parsed_parameter_sets = [mock_param_set]
    return mock_feature_def


@pytest.fixture
def mock_disabled_feature_definition(mock_param_set) -> FeatureDefinition:
    """Create a disabled mock feature definition."""
    with patch(
        "drl_trading_common.config.feature_config_factory.FeatureConfigFactory"
    ) as mock_factory_class:
        mock_factory_instance = MagicMock(spec=FeatureFactoryInterface)
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.get_config_class.return_value = MockParameterSetConfig

        mock_feature_def = FeatureDefinition(
            name="DisabledFeature",
            enabled=False,
            derivatives=[],
            parameter_sets=[],
        )

        mock_feature_def.parsed_parameter_sets = [mock_param_set]
        return mock_feature_def


@pytest.fixture
def mock_features_config(mock_feature_definition: FeatureDefinition) -> FeaturesConfig:
    """Create a mock features configuration."""
    return FeaturesConfig(feature_definitions=[mock_feature_definition])


@pytest.fixture
def mock_mixed_features_config(
    mock_feature_definition: FeatureDefinition,
    mock_disabled_feature_definition: FeatureDefinition,
) -> FeaturesConfig:
    """Create a mixed features configuration with enabled and disabled features."""
    return FeaturesConfig(
        feature_definitions=[mock_feature_definition, mock_disabled_feature_definition]
    )


@pytest.fixture
def mock_historical_data() -> DataFrame:
    """Create mock historical price data."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="H")
    data = {
        "Open": [100 + i * 0.1 for i in range(50)],
        "High": [102 + i * 0.1 for i in range(50)],
        "Low": [98 + i * 0.1 for i in range(50)],
        "Close": [101 + i * 0.1 for i in range(50)],
        "Volume": [1000 + i * 10 for i in range(50)],
    }
    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def mock_current_record() -> Series:
    """Create a mock current record for real-time processing."""
    timestamp = pd.Timestamp("2023-01-01 12:00:00")
    return Series(
        {
            "Time": timestamp,
            "Open": 105.0,
            "High": 106.0,
            "Low": 104.0,
            "Close": 105.5,
            "Volume": 1500,
        },
        name=timestamp,
    )


@pytest.fixture
def mock_asset_data(mock_historical_data) -> AssetPriceDataSet:
    """Create a mock asset price dataset."""
    return AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=mock_historical_data,
    )


@pytest.fixture
def mock_symbol() -> str:
    """Create a mock symbol for testing."""
    return "EURUSD"


@pytest.fixture
def mock_timeframe() -> str:
    """Create a mock timeframe for testing."""
    return "H1"


@pytest.fixture
def mock_feature_factory() -> FeatureFactoryInterface:
    """Create a mock feature factory."""
    factory = MagicMock(spec=FeatureFactoryInterface)

    # Mock the create_feature method to return appropriate instances
    def create_feature_side_effect(feature_name, *args, **kwargs):
        if feature_name == "MockRealTimeFeature":
            return MockRealTimeFeature(*args, **kwargs)
        elif feature_name == "DisabledFeature":
            return MockRealTimeFeature(*args, **kwargs)
        return None

    factory.create_feature.side_effect = create_feature_side_effect
    return factory


@pytest.fixture
def mock_feast_service() -> FeastServiceInterface:
    """Create a mock FeastService."""
    feast_service = MagicMock(spec=FeastService)
    feast_service.is_enabled.return_value = True
    feast_service.get_historical_features.return_value = None
    feast_service.get_online_features.return_value = {}
    return feast_service


@pytest.fixture
def real_time_aggregator(
    mock_features_config, mock_feature_factory, mock_feast_service
) -> RealTimeFeatureAggregator:
    """Create a RealTimeFeatureAggregator instance with mocked dependencies."""
    return RealTimeFeatureAggregator(
        config=mock_features_config,
        feature_factory=mock_feature_factory,
        feast_service=mock_feast_service,
    )


# TDD Test Cases


class TestRealTimeFeatureAggregatorInterface:
    """Test that RealTimeFeatureAggregator implements the interface correctly."""

    def test_implements_interface(self, real_time_aggregator):
        """Test that RealTimeFeatureAggregator implements RealTimeFeatureAggregatorInterface."""
        assert isinstance(real_time_aggregator, RealTimeFeatureAggregatorInterface)

    def test_has_required_methods(self, real_time_aggregator):
        """Test that all interface methods are implemented."""
        assert hasattr(real_time_aggregator, "compute_features_for_single_record")
        assert hasattr(real_time_aggregator, "get_required_lookback_periods")
        assert hasattr(real_time_aggregator, "warm_up_cache")

    def test_method_signatures(self, real_time_aggregator):
        """Test that methods have correct signatures."""
        # Test compute_features_for_single_record signature
        import inspect

        sig = inspect.signature(real_time_aggregator.compute_features_for_single_record)
        expected_params = [
            "current_record",
            "historical_context",
            "symbol",
            "timeframe",
        ]
        actual_params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert actual_params == expected_params

        # Test get_required_lookback_periods signature
        sig = inspect.signature(real_time_aggregator.get_required_lookback_periods)
        assert len(sig.parameters) == 1  # Only 'self'

        # Test warm_up_cache signature
        sig = inspect.signature(real_time_aggregator.warm_up_cache)
        expected_params = ["asset_data", "symbol"]
        actual_params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert actual_params == expected_params


class TestGetRequiredLookbackPeriods:
    """Test the get_required_lookback_periods method."""

    def test_returns_positive_integer(self, real_time_aggregator):
        """Test that lookback periods returns a positive integer."""
        lookback = real_time_aggregator.get_required_lookback_periods()
        assert isinstance(lookback, int)
        assert lookback > 0

    def test_consistent_lookback_value(self, real_time_aggregator):
        """Test that lookback periods returns consistent value."""
        lookback1 = real_time_aggregator.get_required_lookback_periods()
        lookback2 = real_time_aggregator.get_required_lookback_periods()
        assert lookback1 == lookback2

    def test_lookback_with_empty_config(self, mock_feature_factory, mock_feast_service):
        """Test lookback periods with empty feature configuration."""
        empty_config = FeaturesConfig(feature_definitions=[])
        aggregator = RealTimeFeatureAggregator(
            config=empty_config,
            feature_factory=mock_feature_factory,
            feast_service=mock_feast_service,
        )
        lookback = aggregator.get_required_lookback_periods()
        assert isinstance(lookback, int)
        assert lookback >= 10  # Should have a sensible default


class TestWarmUpCache:
    """Test the warm_up_cache method."""

    def test_warm_up_cache_basic(
        self, real_time_aggregator, mock_asset_data, mock_symbol
    ):
        """Test basic cache warm-up functionality."""
        # Should not raise an exception
        real_time_aggregator.warm_up_cache(mock_asset_data, mock_symbol)

    def test_warm_up_cache_initializes_cache(
        self, real_time_aggregator, mock_asset_data, mock_symbol
    ):
        """Test that cache warm-up initializes internal cache."""
        # Initially cache should be empty
        assert len(real_time_aggregator._feature_cache) == 0

        # Warm up cache
        real_time_aggregator.warm_up_cache(mock_asset_data, mock_symbol)

        # Cache should now contain entries (this might fail initially - that's TDD!)
        # We expect at least one cache entry for the enabled feature
        assert len(real_time_aggregator._feature_cache) > 0

    def test_warm_up_cache_with_empty_data(self, real_time_aggregator, mock_symbol):
        """Test cache warm-up with empty asset data."""
        empty_data = AssetPriceDataSet(
            timeframe="H1",
            base_dataset=True,
            asset_price_dataset=DataFrame(),
        )

        # Should handle empty data gracefully
        real_time_aggregator.warm_up_cache(empty_data, mock_symbol)

    def test_warm_up_cache_with_insufficient_data(
        self, real_time_aggregator, mock_symbol
    ):
        """Test cache warm-up with insufficient historical data."""
        # Create minimal data (less than required lookback)
        dates = pd.date_range(start="2023-01-01", periods=5, freq="H")
        minimal_data = DataFrame(
            {
                "Open": [100] * 5,
                "High": [102] * 5,
                "Low": [98] * 5,
                "Close": [101] * 5,
                "Volume": [1000] * 5,
            },
            index=dates,
        )
        minimal_data.index.name = "Time"

        asset_data = AssetPriceDataSet(
            timeframe="H1",
            base_dataset=True,
            asset_price_dataset=minimal_data,
        )

        # Should handle insufficient data gracefully
        real_time_aggregator.warm_up_cache(asset_data, mock_symbol)


class TestComputeFeaturesForSingleRecord:
    """Test the compute_features_for_single_record method."""

    def test_compute_features_basic(
        self,
        real_time_aggregator,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test basic single record feature computation."""
        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        assert isinstance(result, dict)
        # Initially might be empty - that's what TDD catches!

    def test_compute_features_returns_expected_features(
        self,
        real_time_aggregator,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test that computed features match expected feature names."""
        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should contain features from our mock feature
        # This test will initially fail - guiding development
        expected_features = ["rt_feature_1", "rt_feature_2"]
        for feature_name in expected_features:
            assert any(
                feature_name in key for key in result.keys()
            ), f"Missing feature: {feature_name}"

    def test_compute_features_with_empty_historical_context(
        self, real_time_aggregator, mock_current_record, mock_symbol, mock_timeframe
    ):
        """Test computation with empty historical context."""
        empty_df = DataFrame()

        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=empty_df,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

    def test_compute_features_with_insufficient_context(
        self, real_time_aggregator, mock_current_record, mock_symbol, mock_timeframe
    ):
        """Test computation with insufficient historical context."""
        # Create insufficient context (less than 10 rows)
        dates = pd.date_range(start="2023-01-01", periods=5, freq="H")
        insufficient_data = DataFrame(
            {
                "Open": [100] * 5,
                "High": [102] * 5,
                "Low": [98] * 5,
                "Close": [101] * 5,
                "Volume": [1000] * 5,
            },
            index=dates,
        )
        insufficient_data.index.name = "Time"

        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=insufficient_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_compute_features_with_disabled_features(
        self,
        mock_mixed_features_config,
        mock_feature_factory,
        mock_feast_service,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test that disabled features are skipped."""
        aggregator = RealTimeFeatureAggregator(
            config=mock_mixed_features_config,
            feature_factory=mock_feature_factory,
            feast_service=mock_feast_service,
        )

        result = aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should only contain features from enabled feature definitions
        # DisabledFeature should not appear in results
        assert isinstance(result, dict)
        for key in result.keys():
            assert "DisabledFeature" not in key

    def test_compute_features_returns_numeric_values(
        self,
        real_time_aggregator,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test that computed features return numeric values."""
        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        for feature_name, feature_value in result.items():
            assert isinstance(
                feature_value, (int, float)
            ), f"Feature {feature_name} is not numeric: {type(feature_value)}"
            assert not pd.isna(feature_value), f"Feature {feature_name} is NaN"


class TestCacheManagement:
    """Test internal cache management functionality."""

    def test_cache_initialization(self, real_time_aggregator):
        """Test that cache is properly initialized."""
        assert hasattr(real_time_aggregator, "_feature_cache")
        assert isinstance(real_time_aggregator._feature_cache, dict)

    def test_cache_persistence_across_calls(
        self,
        real_time_aggregator,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test that cache persists across multiple computation calls."""
        # First call
        result1 = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        cache_size_after_first = len(real_time_aggregator._feature_cache)

        # Second call with different current record
        modified_record = mock_current_record.copy()
        modified_record["Close"] = 110.0

        result2 = real_time_aggregator.compute_features_for_single_record(
            current_record=modified_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        cache_size_after_second = len(real_time_aggregator._feature_cache)

        # Cache should not grow (features already cached) or only grow slightly
        assert cache_size_after_second >= cache_size_after_first

    def test_get_cached_feature_state(self, real_time_aggregator):
        """Test the get_cached_feature_state method."""
        state = real_time_aggregator.get_cached_feature_state()
        assert isinstance(state, dict)


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_missing_feature_class(
        self,
        mock_features_config,
        mock_feast_service,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,    ):
        """Test behavior when feature class is missing from registry."""
        empty_factory = MagicMock(spec=FeatureFactoryInterface)
        empty_factory.create_feature.return_value = None  # No feature instances

        aggregator = RealTimeFeatureAggregator(
            config=mock_features_config,
            feature_factory=empty_factory,
            feast_service=mock_feast_service,
        )

        result = aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should handle gracefully and return empty dict
        assert isinstance(result, dict)

    def test_invalid_current_record(
        self, real_time_aggregator, mock_historical_data, mock_symbol, mock_timeframe
    ):
        """Test behavior with invalid current record."""
        invalid_record = Series()  # Empty series

        result = real_time_aggregator.compute_features_for_single_record(
            current_record=invalid_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_none_inputs(self, real_time_aggregator, mock_symbol, mock_timeframe):
        """Test behavior with None inputs."""
        result = real_time_aggregator.compute_features_for_single_record(
            current_record=None,
            historical_context=None,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should handle gracefully without raising exceptions
        assert isinstance(result, dict)


class TestIntegrationWithFeastService:
    """Test integration with Feast service."""
    def test_feast_service_disabled(
        self,
        mock_features_config,
        mock_feature_factory,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test behavior when Feast service is disabled."""
        disabled_feast = MagicMock(spec=FeastService)
        disabled_feast.is_enabled.return_value = False

        aggregator = RealTimeFeatureAggregator(
            config=mock_features_config,
            feature_factory=mock_feature_factory,
            feast_service=disabled_feast,
        )

        result = aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        # Should still work without Feast
        assert isinstance(result, dict)

    def test_feast_service_online_features(
        self,
        real_time_aggregator,
        mock_feast_service,
        mock_current_record,
        mock_historical_data,
        mock_symbol,
        mock_timeframe,
    ):
        """Test interaction with Feast online features."""
        # Mock online features response
        mock_feast_service.get_online_features.return_value = {
            "cached_feature_1": 1.5,
            "cached_feature_2": 2.5,
        }

        result = real_time_aggregator.compute_features_for_single_record(
            current_record=mock_current_record,
            historical_context=mock_historical_data,
            symbol=mock_symbol,
            timeframe=mock_timeframe,
        )

        assert isinstance(result, dict)
        # Verify that Feast service was called
        # Note: This depends on implementation details that may not exist yet
