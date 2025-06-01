import copy

import pytest
from drl_trading_common.config.application_config import ApplicationConfig
from injector import Injector

from drl_trading_core.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)


@pytest.fixture
def sample_csv_service(mocked_container: Injector) -> CsvDataImportService:
    """Fixture for testing CSV import."""
    return mocked_container.get(CsvDataImportService)


@pytest.fixture
def create_csv_service_with_limit(mocked_container: Injector):
    """Creates a CSV service with a specific row limit in its configuration.

    Returns:
        Factory function that creates CsvDataImportService with specified limit
    """

    def _factory(limit: int) -> CsvDataImportService:
        # Get the base config from the container
        base_config = mocked_container.get(ApplicationConfig).local_data_import_config

        # Create a deep copy to avoid modifying the original
        modified_config = copy.deepcopy(base_config)

        # Set the new limit
        modified_config.limit = limit

        # Create a new service with the modified config
        return CsvDataImportService(modified_config)

    return _factory


def test_csv_import_basic(create_csv_service_with_limit):
    """Test CSV import functionality with basic validation."""
    # Given
    limit = 1
    expected_symbol = "EURUSD"
    expected_timeframe = "H1"

    # Create a service with the specific limit in config
    csv_service = create_csv_service_with_limit(limit)

    # When
    symbol_containers = csv_service.import_data()

    # Then
    assert len(symbol_containers) > 0, "Expected at least one symbol container"
    symbol_container = symbol_containers[0]
    assert (
        symbol_container.symbol == expected_symbol
    ), f"Expected symbol to be {expected_symbol}"
    assert len(symbol_container.datasets) > 0, "Expected at least one dataset"
    h1_datasets = [
        d for d in symbol_container.datasets if d.timeframe == expected_timeframe
    ]
    assert (
        len(h1_datasets) > 0
    ), f"Expected at least one dataset with timeframe {expected_timeframe}"
    assert not h1_datasets[
        0
    ].asset_price_dataset.empty, f"{expected_timeframe} dataset should not be empty"


def test_csv_import_all_timeframes(create_csv_service_with_limit):
    """Test CSV import functionality with validation of multiple timeframes."""
    # Given
    expected_symbol = "EURUSD"
    expected_timeframes = ["H1", "H4"]
    csv_service = create_csv_service_with_limit(None)

    # When
    symbol_containers = csv_service.import_data()  # No limit to get all data

    # Then
    assert len(symbol_containers) > 0, "Expected at least one symbol container"

    # Find the container for expected_symbol
    eurusd_container = next(
        (c for c in symbol_containers if c.symbol == expected_symbol), None
    )
    assert eurusd_container is not None, f"Container for {expected_symbol} not found"

    # Verify all expected timeframes are present
    timeframes_found = [dataset.timeframe for dataset in eurusd_container.datasets]
    for timeframe in expected_timeframes:
        assert (
            timeframe in timeframes_found
        ), f"Expected timeframe {timeframe} not found"

    # Verify datasets have data
    for timeframe in expected_timeframes:
        dataset = next(
            (d for d in eurusd_container.datasets if d.timeframe == timeframe), None
        )
        assert dataset is not None, f"Dataset for timeframe {timeframe} not found"
        assert (
            not dataset.asset_price_dataset.empty
        ), f"Dataset for timeframe {timeframe} should not be empty"

        # Check if dataset has the expected columns
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for column in expected_columns:
            assert (
                column in dataset.asset_price_dataset.columns
            ), f"Column {column} not found in {timeframe} dataset"


def test_csv_import_with_limit(create_csv_service_with_limit):
    """Test CSV import functionality with different limits in configuration."""
    # Given
    limits = [1, 10, 100]

    for limit in limits:
        # Create a service with the specific limit in config
        csv_service_with_limit = create_csv_service_with_limit(limit)

        # When
        symbol_containers = csv_service_with_limit.import_data()

        # Then
        for container in symbol_containers:
            for dataset in container.datasets:
                # The number of rows should not exceed the limit
                assert (
                    len(dataset.asset_price_dataset) <= limit
                ), f"Dataset exceeds limit of {limit} rows"
