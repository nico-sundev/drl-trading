"""
Unit tests for OfflineFeatureS3Repo.

Tests the S3 implementation of offline feature storage with mocked dependencies
to isolate business logic from external S3 infrastructure.
"""

from io import BytesIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from drl_trading_common.config.feature_config import FeatureStoreConfig
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo import (
    OfflineFeatureS3Repo,
    S3StorageException,
)


class TestOfflineFeatureS3RepoInit:
    """Test class for OfflineFeatureS3Repo initialization."""

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client')
    def test_init_with_valid_config(
        self,
        mock_boto_client: Mock,
        feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        repo = OfflineFeatureS3Repo(feature_store_config)

        # Then
        assert repo.config == feature_store_config
        assert repo.bucket_name == 'drl-trading-features'  # Default value
        assert repo.s3_prefix == 'features'  # Default value
        assert repo._s3_client == mock_s3_client
        mock_s3_client.head_bucket.assert_called_once_with(Bucket='drl-trading-features')

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client')
    def test_init_with_custom_s3_config(
        self,
        mock_boto_client: Mock
    ) -> None:
        """Test initialization with custom S3 configuration."""
        # Given
        config = FeatureStoreConfig(
            enabled=True,
            repo_path="/test/repo",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )
        # Add S3-specific attributes
        config.s3_bucket_name = "custom-bucket"
        config.s3_prefix = "custom-prefix"
        config.s3_endpoint_url = "http://localhost:9000"
        config.s3_access_key_id = "test-key"
        config.s3_secret_access_key = "test-secret"
        config.s3_region = "us-west-2"

        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        repo = OfflineFeatureS3Repo(config)

        # Then
        assert repo.bucket_name == "custom-bucket"
        assert repo.s3_prefix == "custom-prefix"
        mock_boto_client.assert_called_once_with(
            service_name="s3",
            region_name="us-west-2",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client')
    def test_init_bucket_creation_on_404(
        self,
        mock_boto_client: Mock,
        feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test bucket creation when bucket doesn't exist."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client

        # Simulate bucket not found, then successful creation
        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        mock_s3_client.create_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        repo = OfflineFeatureS3Repo(feature_store_config)

        # Then
        assert repo._s3_client == mock_s3_client
        mock_s3_client.head_bucket.assert_called_once_with(Bucket='drl-trading-features')
        mock_s3_client.create_bucket.assert_called_once_with(Bucket='drl-trading-features')

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client')
    def test_init_no_credentials_error(
        self,
        mock_boto_client: Mock,
        feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test error handling when no AWS credentials are available."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.side_effect = NoCredentialsError()

        # When & Then
        with pytest.raises(S3StorageException, match="No AWS credentials found"):
            OfflineFeatureS3Repo(feature_store_config)


class TestOfflineFeatureS3RepoStoreIncremental:
    """Test class for incremental feature storage operations."""

    @pytest.fixture
    def mock_s3_repo(self, feature_store_config: FeatureStoreConfig) -> OfflineFeatureS3Repo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client'):
            repo = OfflineFeatureS3Repo(feature_store_config)
            repo._s3_client = Mock()
            return repo

    def test_store_features_incrementally_new_dataset(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features for a new dataset with no existing data."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}  # No existing objects
        mock_s3_repo._s3_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        stored_count = mock_s3_repo.store_features_incrementally(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == len(sample_features_df)
        # Verify S3 client was called to store data
        assert mock_s3_repo._s3_client.put_object.called

    def test_store_features_incrementally_with_duplicates(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features when some timestamps already exist."""
        # Given
        # Mock existing features response
        existing_parquet_bytes = self._create_parquet_bytes(sample_features_df.iloc[:2])  # First 2 rows exist
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "features/EURUSD/existing.parquet"}]
        }
        mock_s3_repo._s3_client.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=existing_parquet_bytes))
        }

        # Create overlapping dataset with some new data
        overlapping_df = sample_features_df.copy()
        new_timestamp = pd.Timestamp("2024-01-05 10:00:00")
        new_row = overlapping_df.iloc[-1:].copy()
        new_row["event_timestamp"] = new_timestamp
        new_row["feature_1"] = 999.0
        overlapping_df = pd.concat([overlapping_df, new_row], ignore_index=True)

        # When
        stored_count = mock_s3_repo.store_features_incrementally(
            overlapping_df,
            eurusd_h1_symbol
        )

        # Then
        # Should only store the new records (last 2 from original + 1 new)
        expected_new_count = len(sample_features_df) - 2 + 1
        assert stored_count == expected_new_count

    def test_store_features_incrementally_missing_timestamp_column(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0]
        })

        # When & Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            mock_s3_repo.store_features_incrementally(invalid_df, eurusd_h1_symbol)

    def test_store_features_incrementally_empty_dataframe(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test handling of empty DataFrame during storage."""
        # Given
        empty_df = DataFrame({
            "event_timestamp": []
        })

        # When
        stored_count = mock_s3_repo.store_features_incrementally(
            empty_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == 0
        # Verify no S3 operations were performed
        assert not mock_s3_repo._s3_client.put_object.called

    def test_store_features_incrementally_s3_error(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when S3 operations fail."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"
        )

        # When & Then
        with pytest.raises(S3StorageException, match="Failed to store features incrementally"):
            mock_s3_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()


class TestOfflineFeatureS3RepoLoadExisting:
    """Test class for loading existing features."""

    @pytest.fixture
    def mock_s3_repo(self, feature_store_config: FeatureStoreConfig) -> OfflineFeatureS3Repo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client'):
            repo = OfflineFeatureS3Repo(feature_store_config)
            repo._s3_client = Mock()
            return repo

    def test_load_existing_features_no_data(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when no data exists."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}  # No objects

        # When
        loaded_features = mock_s3_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert loaded_features is None

    def test_load_existing_features_with_data(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when data exists."""
        # Given
        parquet_bytes = self._create_parquet_bytes(sample_features_df)
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "features/EURUSD/test.parquet"}]
        }
        mock_s3_repo._s3_client.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=parquet_bytes))
        }

        # When
        loaded_features = mock_s3_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert loaded_features is not None
        assert len(loaded_features) == len(sample_features_df)
        pd.testing.assert_frame_equal(
            loaded_features.sort_values("event_timestamp").reset_index(drop=True),
            sample_features_df.sort_values("event_timestamp").reset_index(drop=True)
        )

    def test_load_existing_features_multiple_files(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features from multiple S3 objects."""
        # Given
        df1 = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 10:00:00")],
            "feature_1": [1.0]
        })
        df2 = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-02 10:00:00")],
            "feature_1": [2.0]
        })

        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "features/EURUSD/file1.parquet"},
                {"Key": "features/EURUSD/file2.parquet"}
            ]
        }

        mock_s3_repo._s3_client.get_object.side_effect = [
            {"Body": Mock(read=Mock(return_value=self._create_parquet_bytes(df1)))},
            {"Body": Mock(read=Mock(return_value=self._create_parquet_bytes(df2)))}
        ]

        # When
        loaded_features = mock_s3_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert loaded_features is not None
        assert len(loaded_features) == 2
        assert len(mock_s3_repo._s3_client.get_object.call_args_list) == 2

    def test_load_existing_features_s3_error(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when S3 operations fail during loading."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"
        )

        # When & Then
        with pytest.raises(S3StorageException, match="Failed to load existing features"):
            mock_s3_repo.load_existing_features(eurusd_h1_symbol)

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()


class TestOfflineFeatureS3RepoUtilityMethods:
    """Test class for utility methods."""

    @pytest.fixture
    def mock_s3_repo(self, feature_store_config: FeatureStoreConfig) -> OfflineFeatureS3Repo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client'):
            repo = OfflineFeatureS3Repo(feature_store_config)
            repo._s3_client = Mock()
            return repo

    def test_feature_exists_false_for_new_symbol(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns False for new symbol."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}  # No objects

        # When
        exists = mock_s3_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert not exists

    def test_feature_exists_true_after_storage(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns True when objects exist."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "features/EURUSD/test.parquet"}]
        }

        # When
        exists = mock_s3_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert exists

    def test_get_feature_count_zero_for_new_symbol(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns 0 for new symbol."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}  # No objects

        # When
        count = mock_s3_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == 0

    def test_get_feature_count_correct_after_storage(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns correct count when features exist."""
        # Given
        parquet_bytes = self._create_parquet_bytes(sample_features_df)
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "features/EURUSD/test.parquet"}]
        }
        mock_s3_repo._s3_client.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=parquet_bytes))
        }

        # When
        count = mock_s3_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == len(sample_features_df)

    def test_delete_features_success(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test successful deletion of features."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "features/EURUSD/file1.parquet"},
                {"Key": "features/EURUSD/file2.parquet"}
            ]
        }
        mock_s3_repo._s3_client.delete_objects.return_value = {
            "Deleted": [
                {"Key": "features/EURUSD/file1.parquet"},
                {"Key": "features/EURUSD/file2.parquet"}
            ]
        }

        # When
        result = mock_s3_repo.delete_features(eurusd_h1_symbol)

        # Then
        assert result is True
        mock_s3_repo._s3_client.delete_objects.assert_called_once()

    def test_delete_features_no_objects(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test deletion when no objects exist."""
        # Given
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}  # No objects

        # When
        result = mock_s3_repo.delete_features(eurusd_h1_symbol)

        # Then
        assert result is False
        assert not mock_s3_repo._s3_client.delete_objects.called

    def test_get_storage_metrics(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storage metrics retrieval."""
        # Given
        from datetime import datetime
        last_modified = datetime(2024, 1, 1, 12, 0, 0)

        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "features/EURUSD/file1.parquet", "Size": 1024, "LastModified": last_modified},
                {"Key": "features/EURUSD/file2.parquet", "Size": 2048, "LastModified": last_modified}
            ]
        }

        # When
        metrics = mock_s3_repo.get_storage_metrics(eurusd_h1_symbol)

        # Then
        assert metrics["size_bytes"] == 3072  # 1024 + 2048
        assert metrics["object_count"] == 2
        assert metrics["last_modified"] == last_modified

    def test_store_features_batch(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame
    ) -> None:
        """Test batch storage of multiple feature datasets."""
        # Given
        batch_data = [
            {"symbol": "EURUSD", "features_df": sample_features_df.iloc[:2]},
            {"symbol": "GBPUSD", "features_df": sample_features_df.iloc[2:]}
        ]

        # Mock no existing data for both symbols
        mock_s3_repo._s3_client.list_objects_v2.return_value = {}
        mock_s3_repo._s3_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        results = mock_s3_repo.store_features_batch(batch_data)

        # Then
        assert results["EURUSD"] == 2
        assert results["GBPUSD"] == len(sample_features_df) - 2

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()


class TestOfflineFeatureS3RepoErrorHandling:
    """Test class for error handling scenarios."""

    @pytest.fixture
    def mock_s3_repo(self, feature_store_config: FeatureStoreConfig) -> OfflineFeatureS3Repo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo.boto3.client'):
            repo = OfflineFeatureS3Repo(feature_store_config)
            repo._s3_client = Mock()
            return repo

    def test_schema_validation_failure(
        self,
        mock_s3_repo: OfflineFeatureS3Repo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test schema validation failure with incompatible schemas."""
        # Given
        # Mock existing features with different schema
        existing_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 10:00:00")],
            "different_feature": [1.0]  # Different column name
        })

        parquet_bytes = self._create_parquet_bytes(existing_df)
        mock_s3_repo._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "features/EURUSD/existing.parquet"}]
        }
        mock_s3_repo._s3_client.get_object.return_value = {
            "Body": Mock(read=Mock(return_value=parquet_bytes))
        }

        # Create new features with incompatible schema (missing column)
        incompatible_features = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-05 10:00:00")],
            "new_feature": [999.0]  # Missing 'different_feature' column
        })

        # When & Then
        with pytest.raises(S3StorageException, match="Failed to store features incrementally"):
            mock_s3_repo.store_features_incrementally(incompatible_features, eurusd_h1_symbol)

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()
