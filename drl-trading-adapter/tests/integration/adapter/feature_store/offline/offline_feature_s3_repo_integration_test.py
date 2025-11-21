"""
Integration tests for S3-based offline feature storage using TestContainers.

These tests use real S3-compatible storage (MinIO) to validate the S3 implementation
against actual S3 API behavior, providing much more reliable testing than mocking.
"""

import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.offline.parquet import OfflineS3ParquetFeatureRepo, S3StorageException


class TestOfflineS3ParquetFeatureRepoIntegration:
    """Integration tests for S3-based offline feature repository."""

    def test_store_features_incrementally_new_dataset(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features for a new dataset with no existing S3 data."""
        # When
        stored_count = offline_s3_repo.store_features_incrementally(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == len(sample_features_df)

    def test_store_features_incrementally_with_duplicates(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features when some timestamps already exist in S3."""
        # Given
        # Store initial features to S3
        offline_s3_repo.store_features_incrementally(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Create overlapping dataset with some new data
        overlapping_df = sample_features_df.copy()
        new_timestamp = pd.Timestamp("2024-01-05 10:00:00")
        new_row = overlapping_df.iloc[-1:].copy()
        new_row["event_timestamp"] = new_timestamp
        new_row["feature_1"] = 999.0
        overlapping_df = pd.concat([overlapping_df, new_row], ignore_index=True)

        # When
        stored_count = offline_s3_repo.store_features_incrementally(
            overlapping_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == 1  # Only the new row should be stored

    def test_get_repo_path(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test getting the repository path for a symbol."""
        # When
        repo_path = offline_s3_repo.get_repo_path(eurusd_h1_symbol)

        # Then
        assert repo_path.startswith("s3://")
        assert eurusd_h1_symbol in repo_path

    def test_get_repo_path_invalid_symbol(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo
    ) -> None:
        """Test get_repo_path with invalid symbol."""
        # When & Then
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            offline_s3_repo.get_repo_path("")

    def test_s3_error_handling_network_failure(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when S3 operations fail."""
        # Given
        # Simulate network failure by corrupting S3 bucket name
        original_bucket = offline_s3_repo.bucket_name
        offline_s3_repo.bucket_name = "invalid-nonexistent-bucket-12345"

        # When & Then
        with pytest.raises(S3StorageException):  # Should raise S3-specific exception
            offline_s3_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Cleanup
        offline_s3_repo.bucket_name = original_bucket

    def test_large_dataset_s3_performance(
        self,
        offline_s3_repo: OfflineS3ParquetFeatureRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test performance with larger datasets in S3."""
        # Given
        # Create larger dataset (1000 records)
        timestamps = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=1000,
            freq="h"
        )
        large_features_df = DataFrame({
            "event_timestamp": timestamps,
            "symbol": [eurusd_h1_symbol] * len(timestamps),
            "feature_1": [30.5 + i * 0.1 for i in range(len(timestamps))],
            "feature_2": [1.0850 + i * 0.00001 for i in range(len(timestamps))],
        })

        # When
        stored_count = offline_s3_repo.store_features_incrementally(
            large_features_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == len(large_features_df)

    def test_s3_bucket_permissions(
        self,
        s3_client_minio,
        s3_test_bucket: str,
        offline_s3_repo: OfflineS3ParquetFeatureRepo
    ) -> None:
        """Test S3 bucket permissions and access patterns."""
        # Given
        # Test bucket should be accessible
        bucket_response = s3_client_minio.head_bucket(Bucket=s3_test_bucket)
        assert bucket_response["ResponseMetadata"]["HTTPStatusCode"] == 200

        # When & Then
        # Repository should be able to get repo path without error
        repo_path = offline_s3_repo.get_repo_path("TESTSYMBOL")
        assert repo_path is not None
