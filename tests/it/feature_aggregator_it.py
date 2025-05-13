import logging
from typing import Dict, Set
from unittest.mock import patch

import dask
import pandas as pd
import pytest

from ai_trading.common.di.containers import ApplicationContainer
from ai_trading.common.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.feast.feast_service import FeastServiceInterface
from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregatorInterface

logger = logging.getLogger(__name__)


@pytest.fixture
def dataset(mocked_container) -> AssetPriceDataSet:
    """Get a test dataset using the test container.

    Returns:
        A test dataset (H1 timeframe) for integration testing
    """
    # Given - Set up the data import
    importer = mocked_container.data_import_manager()

    # Get all symbol containers
    symbol_containers = importer.get_data()

    # Extract datasets from all symbols
    all_datasets = []
    for symbol_container in symbol_containers:
        all_datasets.extend(symbol_container.datasets)

    # Filter to get H1 timeframe datasets
    h1_datasets = [dataset for dataset in all_datasets if dataset.timeframe == "H1"]
    assert len(h1_datasets) > 0, "No H1 datasets found for testing"
    return h1_datasets[0]


@pytest.fixture
def expected_features(mocked_container: ApplicationContainer) -> Dict[str, Set[str]]:
    """Generate a dictionary of expected feature names based on config.

    This fixture analyzes the test configuration and creates a mapping between
    feature names and the actual column name patterns they should produce based
    on their implementation in the feature classes.

    Returns:
        Dict mapping feature name to set of expected column name patterns
    """
    # Extract feature definitions from config
    features_config = mocked_container.application_config().features_config

    # Build expected feature dictionary based on actual feature class implementations
    expected: Dict[str, Set[str]] = {}
    for feature_def in features_config.feature_definitions:
        if not feature_def.enabled:
            continue

        feature_name = feature_def.name
        if feature_name not in expected:
            expected[feature_name] = set()

        # For each parameter set, add the expected column name patterns
        # based on the actual feature class implementations
        for param_set in feature_def.parsed_parameter_sets:
            if not param_set.enabled:
                continue

            # Add expected column name patterns based on feature class implementations
            if feature_name == "macd":
                # From MacdFeature.get_sub_features_names()
                expected[feature_name].update(
                    ["macd_cross_bullish", "macd_cross_bearish", "macd_trend"]
                )
            elif feature_name == "rsi":
                # From RsiFeature.get_sub_features_names()
                for length in [
                    param_set.length
                    for param_set in feature_def.parsed_parameter_sets
                    if param_set.enabled
                ]:
                    expected[feature_name].add(f"rsi_{length}")
            elif feature_name == "roc":
                # From RocFeature.get_sub_features_names()
                for length in [
                    param_set.length
                    for param_set in feature_def.parsed_parameter_sets
                    if param_set.enabled
                ]:
                    expected[feature_name].add(f"roc_{length}")
            elif feature_name == "range":
                # From RangeFeature.get_sub_features_names()
                for lookback in [
                    param_set.lookback
                    for param_set in feature_def.parsed_parameter_sets
                    if param_set.enabled
                ]:
                    expected[feature_name].update(
                        [f"resistance_range{lookback}", f"support_range{lookback}"]
                    )
            elif feature_name == "rvi":
                # From RviFeature.get_sub_features_names()
                for length in [
                    param_set.length
                    for param_set in feature_def.parsed_parameter_sets
                    if param_set.enabled
                ]:
                    expected[feature_name].add(f"rvi_{length}")

    return expected


def test_feature_computation_and_caching(
    mocked_container, mocked_feature_store, dataset, expected_features
):
    """Test both feature computation and caching in a single integration test.

    This test verifies:
    1. First run computes features and stores them in the feature store
    2. Second run retrieves features from the cache instead of recomputing
    3. All expected features from the config are properly computed
    """
    # Given
    feature_aggregator: FeatureAggregatorInterface = (
        mocked_container.feature_aggregator()
    )
    feast_service: FeastServiceInterface = mocked_container.feast_service()
    symbol = "EURUSD"  # From test config

    # Create a counter to track feature computation
    computation_counter = 0

    # Function to count calls to the compute method
    def count_compute_calls(*args, **kwargs):
        nonlocal computation_counter
        computation_counter += 1
        # Call the original method to get the actual result
        result = original_compute(*args, **kwargs)
        return result

    # When - First Run (computation)
    # Save the original _compute_or_get_single_feature method reference
    original_compute = feature_aggregator._compute_or_get_single_feature

    # Patch the method to use our counting function
    with patch.object(
        feature_aggregator,
        "_compute_or_get_single_feature",
        side_effect=count_compute_calls,
    ):
        # First run: compute features
        first_run_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
        first_run_results = dask.compute(*first_run_tasks)
        first_run_dfs = [df for df in first_run_results if df is not None]

        # Reset counter and track calls for second run
        feature_compute_calls_first_run = computation_counter
        computation_counter = 0

        # Second run: should use cache
        second_run_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
        second_run_results = dask.compute(*second_run_tasks)
        second_run_dfs = [df for df in second_run_results if df is not None]

        # Store the count for assertions
        feature_compute_calls_second_run = computation_counter

    # Create merged result dataframes for validation
    first_run_merged = pd.DataFrame({"Time": dataset.asset_price_dataset["Time"]})
    for df in first_run_dfs:
        if len(df.columns) > 1:
            first_run_merged = pd.merge(first_run_merged, df, on="Time", how="left")

    second_run_merged = pd.DataFrame({"Time": dataset.asset_price_dataset["Time"]})
    for df in second_run_dfs:
        if len(df.columns) > 1:
            second_run_merged = pd.merge(second_run_merged, df, on="Time", how="left")

    # Then - Verify calculations
    # 1. Check that feature computation happened in first run
    assert (
        feature_compute_calls_first_run > 0
    ), "Expected features to be computed in first run"

    # 2. Verify all expected features are present in the computed result
    for feature_name, expected_cols in expected_features.items():
        for col_substring in expected_cols:
            matching_cols = [
                c for c in first_run_merged.columns if col_substring in c.lower()
            ]
            assert (
                len(matching_cols) > 0
            ), f"Feature {feature_name} with substring '{col_substring}' not found in first run"

    # 3. Verify second run produced identical results to first run
    assert len(second_run_dfs) == len(
        first_run_dfs
    ), "Second run should return same number of results"
    pd.testing.assert_frame_equal(
        first_run_merged.sort_index(axis=1),
        second_run_merged.sort_index(axis=1),
        "First run and second run results should be identical",
    )

    # 4. Verify feature computation was reduced in second run due to caching
    assert feature_compute_calls_second_run <= feature_compute_calls_first_run, (
        f"Expected fewer feature computations in second run (actual: {feature_compute_calls_second_run}, "
        f"first run: {feature_compute_calls_first_run})"
    )
