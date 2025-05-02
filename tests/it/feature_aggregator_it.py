import dask
import pytest

from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregatorInterface


@pytest.fixture
def dataset(test_container):
    """Get a test dataset using the test container.

    Args:
        test_container: The test container fixture

    Returns:
        A test dataset for integration testing
    """
    # Given
    importer = test_container.data_import_manager()

    # Get all symbol containers
    symbol_containers = importer.get_data(100)

    # Extract datasets from all symbols
    all_datasets = []
    for symbol_container in symbol_containers:
        all_datasets.extend(symbol_container.datasets)

    # Filter to get H1 timeframe datasets
    h1_datasets = [dataset for dataset in all_datasets if dataset.timeframe == "H1"]
    return h1_datasets[0]


def test_feature_computation(test_container, dataset):
    """Test feature computation using the feature aggregator.

    Args:
        test_container: The test container fixture
        dataset: The test dataset fixture
    """
    # Given
    feature_aggregator: FeatureAggregatorInterface = test_container.feature_aggregator()
    expected_columns = ["Time", "rsi_7"]
    symbol = "EURUSD"  # Assuming this is the symbol for the test dataset

    # When
    # Get delayed tasks from compute
    delayed_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)

    # Execute the delayed tasks using dask.compute
    computed_results = dask.compute(*delayed_tasks)

    # Filter out None results
    computed_dfs = [df for df in computed_results if df is not None]

    # Combine results (simplified for test - just checking the first computed dataframe)
    result_df = computed_dfs[0] if computed_dfs else None

    # Then
    assert result_df is not None
    assert set(expected_columns).issubset(set(result_df.columns))
