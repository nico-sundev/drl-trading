"""
S3 Feature Store Unit Tests with moto (AWS mocking library).

These are focused unit tests that verify S3 storage behavior in isolation
using moto for fast, reliable mocking without Docker dependencies.

Focus Areas:
1. S3 storage operations (save/fetch)
2. Error handling for network issues
3. Data serialization/deserialization
4. Bucket and key management

For full integration testing with realistic S3 scenarios, see:
tests/integration/preprocess/feature_store/s3_feature_store_integration_test.py
"""

import logging
import pytest
import pandas as pd
import boto3
from moto import mock_aws
from pandas import DataFrame

from drl_trading_common.config.feature_config import FeatureStoreConfig, S3RepoConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo import (
    OfflineFeatureS3Repo,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_feature_data() -> DataFrame:
    """Create minimal test data."""
    return pd.DataFrame({
        'symbol': ['EURUSD', 'EURUSD'],
        'timeframe': ['H1', 'H1'],
        'event_timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
        'feature_value': [1.1050, 1.1055],
        'feature_signal': [1, 0]
    })


def create_s3_repo() -> OfflineFeatureS3Repo:
    """Create S3 repository instance for unit testing."""
    s3_repo_config = S3RepoConfig(
        bucket_name="test-unit-bucket",
        prefix="unit-tests",
        region="us-east-1",
        endpoint_url=None,
        access_key_id="testing",
        secret_access_key="testing"
    )

    config = FeatureStoreConfig(
        enabled=True,
        entity_name="unit_test_entity",
        ttl_days=1,  # Minimal for unit tests
        online_enabled=False,
        service_name="unit_test",
        service_version="1.0.0",
        config_directory="/tmp/unit_test",  # Not used in unit tests
        offline_repo_strategy=OfflineRepoStrategyEnum.S3,
        s3_repo_config=s3_repo_config
    )

    return OfflineFeatureS3Repo(config)


class TestS3FeatureStoreUnit:
    """Unit tests for S3 feature store focusing on isolated S3 behavior."""

    @mock_aws
    def test_save_and_fetch_basic_workflow(
        self,
        sample_feature_data: DataFrame
    ) -> None:
        """Test basic save and fetch workflow with mocked S3."""
        # Given
        # Create bucket in mocked S3
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-unit-bucket")

        # Create repository after moto mock is active
        s3_repo = create_s3_repo()

        # When
        symbol = "EURUSD"
        s3_repo.store_features_incrementally(sample_feature_data, symbol)
        retrieved_data = s3_repo.load_existing_features(symbol)

        # Then
        assert retrieved_data is not None
        assert len(retrieved_data) == 2
        pd.testing.assert_frame_equal(
            retrieved_data.sort_values('event_timestamp').reset_index(drop=True),
            sample_feature_data.sort_values('event_timestamp').reset_index(drop=True)
        )

    @mock_aws
    def test_fetch_nonexistent_returns_none(
        self
    ) -> None:
        """Test that fetching non-existent features returns None."""
        # Given
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-unit-bucket")

        # Create repository after moto mock is active
        s3_repo = create_s3_repo()

        # When
        symbol = "EURUSD"
        result = s3_repo.load_existing_features(symbol)

        # Then
        assert result is None

    @mock_aws
    def test_incremental_storage_duplicate_prevention(
        self,
        sample_feature_data: DataFrame
    ) -> None:
        """Test that incremental storage prevents duplicate timestamps."""
        # Given
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-unit-bucket")

        # Create repository after moto mock is active
        s3_repo = create_s3_repo()

        # Create modified data with same timestamps (should be skipped)
        modified_data = sample_feature_data.copy()
        modified_data['feature_value'] = modified_data['feature_value'] * 2

        # When
        symbol = "EURUSD"
        # Save original
        records_stored_1 = s3_repo.store_features_incrementally(sample_feature_data, symbol)

        # Try to store modified data with same timestamps (should be skipped)
        records_stored_2 = s3_repo.store_features_incrementally(modified_data, symbol)

        # Fetch result
        result = s3_repo.load_existing_features(symbol)

        # Then
        assert result is not None
        assert records_stored_1 == 2  # Original data stored
        assert records_stored_2 == 0  # Modified data skipped due to duplicate timestamps
        assert len(result) == 2  # Only original data remains

        # Verify original data is preserved (not overwritten)
        pd.testing.assert_frame_equal(
            result.sort_values('event_timestamp').reset_index(drop=True),
            sample_feature_data.sort_values('event_timestamp').reset_index(drop=True)
        )

    @mock_aws
    def test_large_dataset_handling(
        self
    ) -> None:
        """Test handling of larger datasets for performance verification."""
        # Given
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-unit-bucket")

        # Create repository after moto mock is active
        s3_repo = create_s3_repo()

        # Create larger dataset (but still reasonable for unit test)
        large_data = pd.DataFrame({
            'symbol': ['EURUSD'] * 100,
            'timeframe': ['H1'] * 100,
            'event_timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'feature_value': range(100),
            'feature_signal': [i % 3 - 1 for i in range(100)]
        })

        # When
        symbol = "EURUSD"
        s3_repo.store_features_incrementally(large_data, symbol)
        retrieved_data = s3_repo.load_existing_features(symbol)

        # Then
        assert retrieved_data is not None
        assert len(retrieved_data) == 100
        pd.testing.assert_frame_equal(
            retrieved_data.sort_values('event_timestamp').reset_index(drop=True),
            large_data.sort_values('event_timestamp').reset_index(drop=True)
        )

    @mock_aws
    def test_multiple_feature_isolation(
        self,
        sample_feature_data: DataFrame
    ) -> None:
        """Test that different features don't interfere with each other."""
        # Given
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-unit-bucket")

        # Create repository after moto mock is active
        s3_repo = create_s3_repo()

        data_2 = sample_feature_data.copy()
        data_2['feature_value'] = data_2['feature_value'] * 3

        # When
        symbol_1 = "EURUSD"
        symbol_2 = "GBPUSD"

        s3_repo.store_features_incrementally(sample_feature_data, symbol_1)
        s3_repo.store_features_incrementally(data_2, symbol_2)

        result_1 = s3_repo.load_existing_features(symbol_1)
        result_2 = s3_repo.load_existing_features(symbol_2)

        # Then
        assert result_1 is not None
        assert result_2 is not None

        # Verify isolation - each feature has its own data
        pd.testing.assert_frame_equal(
            result_1.sort_values('event_timestamp').reset_index(drop=True),
            sample_feature_data.sort_values('event_timestamp').reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(
            result_2.sort_values('event_timestamp').reset_index(drop=True),
            data_2.sort_values('event_timestamp').reset_index(drop=True)
        )
