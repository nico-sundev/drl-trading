"""
Unit tests for OfflineS3ParquetFeatureRepo.

Tests the S3 implementation of offline feature storage with mocked dependencies
to isolate business logic from external S3 infrastructure.
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.offline.parquet import OfflineS3ParquetFeatureRepo, S3StorageException
from drl_trading_common.config.feature_config import FeatureStoreConfig, S3RepoConfig


@pytest.fixture
def s3_config():
    """Create an S3 repository configuration."""
    return FeatureStoreConfig(
        offline_repo_strategy="s3",
        s3_repo_config=S3RepoConfig(
            bucket_name="test-bucket",
            s3_key_prefix="test-prefix"
        ),
        cache_enabled=False,
        entity_name="test_entity",
        ttl_days=30,
        service_name="test_service",
        service_version="1.0.0",
        config_directory="/tmp/test"
    )


@pytest.fixture
def mock_s3_client():
    """Create a mocked boto3 S3 client."""
    from botocore.exceptions import ClientError

    client = MagicMock()

    # Mock list_objects_v2 to return empty by default
    client.list_objects_v2.return_value = {"Contents": []}

    # Mock put_object to succeed
    client.put_object.return_value = {}

    # Mock get_object to raise NoSuchKey by default
    def mock_get_object_error(*args, **kwargs):
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
        raise ClientError(error_response, 'GetObject')

    client.get_object.side_effect = mock_get_object_error

    return client


@pytest.fixture
def repo(s3_config, mock_s3_client):
    """Create an OfflineS3ParquetFeatureRepo instance with mocked S3."""
    with patch('boto3.client', return_value=mock_s3_client):
        return OfflineS3ParquetFeatureRepo(s3_config)


@pytest.fixture
def sample_features() -> DataFrame:
    """Create sample feature data."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=100, freq="5min")
    df = DataFrame({
        "event_timestamp": timestamps,
        "symbol": "BTCUSDT",
        "rsi_14": range(100),
        "ema_20": range(100, 200)
    })
    df.index = pd.Index(range(len(df)), dtype='int64')  # Avoid RangeIndex issues
    return df


class TestOfflineS3ParquetFeatureRepoInit:
    """Test class for OfflineS3ParquetFeatureRepo initialization."""

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_with_valid_config(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        repo = OfflineS3ParquetFeatureRepo(s3_feature_store_config)

        # Then
        assert repo.config == s3_feature_store_config
        assert repo.bucket_name == 'test-bucket'  # From S3 config
        assert repo.s3_prefix == 'test-prefix'  # From S3 config
        assert repo._s3_client == mock_s3_client
        mock_s3_client.head_bucket.assert_called_once_with(Bucket='test-bucket')

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_with_custom_s3_config(
        self,
        mock_boto_client: Mock
    ) -> None:
        """Test initialization with custom S3 configuration."""
        # Given
        from drl_trading_common.config.feature_config import S3RepoConfig
        from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum

        s3_config = S3RepoConfig(
            bucket_name="custom-bucket",
            prefix="custom-prefix",
            region="us-west-2",
            endpoint_url="http://localhost:9000",
            access_key_id="test-key",
            secret_access_key="test-secret"
        )

        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.S3,
            s3_repo_config=s3_config
        )

        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        # When
        repo = OfflineS3ParquetFeatureRepo(config)

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

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_bucket_creation_on_404(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
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
        repo = OfflineS3ParquetFeatureRepo(s3_feature_store_config)

        # Then
        assert repo._s3_client == mock_s3_client
        mock_s3_client.head_bucket.assert_called_once_with(Bucket='test-bucket')
        mock_s3_client.create_bucket.assert_called_once_with(Bucket='test-bucket')

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_no_credentials_error(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test error handling when no AWS credentials are available."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.side_effect = NoCredentialsError()

        # When & Then
        with pytest.raises(S3StorageException, match="No AWS credentials found"):
            OfflineS3ParquetFeatureRepo(s3_feature_store_config)

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_missing_s3_config_raises_error(
        self,
        mock_boto_client: Mock
    ) -> None:
        """Test that missing S3 config raises ValueError."""
        # Given
        config = FeatureStoreConfig(
            offline_repo_strategy="s3",
            s3_repo_config=None,  # Missing S3 config
            cache_enabled=False,
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            config_directory="/tmp/test"
        )

        # When & Then
        with pytest.raises(ValueError, match="s3_repo_config is required"):
            OfflineS3ParquetFeatureRepo(config)

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_bucket_access_denied_error(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test error handling when bucket access is denied."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client

        # Simulate access denied error (not 404)
        error_response = {"Error": {"Code": "AccessDenied"}}
        mock_s3_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

        # When & Then
        with pytest.raises(S3StorageException, match="Cannot access S3 bucket"):
            OfflineS3ParquetFeatureRepo(s3_feature_store_config)

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_bucket_creation_failure(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test error handling when bucket creation fails."""
        # Given
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client

        # Simulate bucket not found, then creation failure
        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        mock_s3_client.create_bucket.side_effect = Exception("Creation failed")

        # When & Then
        with pytest.raises(S3StorageException, match="could not be created"):
            OfflineS3ParquetFeatureRepo(s3_feature_store_config)

    @patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client')
    def test_init_s3_client_creation_failure(
        self,
        mock_boto_client: Mock,
        s3_feature_store_config: FeatureStoreConfig
    ) -> None:
        """Test error handling when S3 client creation fails."""
        # Given
        mock_boto_client.side_effect = Exception("S3 client creation failed")

        # When & Then
        with pytest.raises(S3StorageException, match="Failed to initialize S3 client"):
            OfflineS3ParquetFeatureRepo(s3_feature_store_config)


class TestOfflineS3ParquetFeatureRepoStoreIncremental:
    """Test class for incremental feature storage operations."""

    @pytest.fixture
    def mock_s3_repo(self, s3_feature_store_config: FeatureStoreConfig) -> OfflineS3ParquetFeatureRepo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client'):
            repo = OfflineS3ParquetFeatureRepo(s3_feature_store_config)
            repo._s3_client = Mock()
            return repo

    def test_store_features_incrementally_new_dataset(
        self,
        mock_s3_repo: OfflineS3ParquetFeatureRepo,
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

    def test_store_features_incrementally_missing_timestamp_column(
        self,
        mock_s3_repo: OfflineS3ParquetFeatureRepo,
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
        mock_s3_repo: OfflineS3ParquetFeatureRepo,
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

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()


class TestOfflineS3ParquetFeatureRepoErrorHandling:
    """Test class for error handling scenarios."""

    @pytest.fixture
    def mock_s3_repo(self, s3_feature_store_config: FeatureStoreConfig) -> OfflineS3ParquetFeatureRepo:
        """Create a mocked S3 repository for testing."""
        with patch('drl_trading_adapter.adapter.feature_store.offline.parquet.offline_s3_parquet_feature_repo.boto3.client'):
            repo = OfflineS3ParquetFeatureRepo(s3_feature_store_config)
            repo._s3_client = Mock()
            return repo

    def _create_parquet_bytes(self, df: DataFrame) -> bytes:
        """Helper method to create parquet bytes from DataFrame."""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()


class TestStoreFeaturesIncrementally:
    """Test suite for incremental storage operations."""

    def test_store_initial_features(self, repo, mock_s3_client, sample_features):
        """Test storing features for the first time."""
        count = repo.store_features_incrementally(sample_features, "BTCUSDT")

        assert count == 100

        # Verify S3 put_object was called for parquet files
        put_calls = [call for call in mock_s3_client.put_object.call_args_list
                     if call[1]['Key'].endswith('.parquet')]
        assert len(put_calls) > 0

    def test_store_non_overlapping_features(self, repo, mock_s3_client, sample_features):
        """Test storing new features that don't overlap with existing ones."""
        # Mock metadata showing existing data
        existing_metadata = {
            "partitions": [{
                "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/features_20240101_000000_081500.parquet",
                "min_timestamp": "2024-01-01 00:00:00",
                "max_timestamp": "2024-01-01 08:15:00",
                "record_count": 100
            }]
        }

        def mock_get_object(Bucket, Key):
            if "_metadata.json" in Key:
                return {
                    "Body": BytesIO(json.dumps(existing_metadata).encode('utf-8'))
                }
            raise Exception("NoSuchKey")

        mock_s3_client.get_object.side_effect = mock_get_object

        # Store non-overlapping features [10:00 - 18:15]
        new_timestamps = pd.date_range("2024-01-01 10:00:00", periods=100, freq="5min")
        new_features = DataFrame({
            "event_timestamp": new_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(100, 200),
            "ema_20": range(200, 300)
        })
        new_features.index = pd.Index(range(len(new_features)), dtype='int64')  # Avoid RangeIndex issues

        count = repo.store_features_incrementally(new_features, "BTCUSDT")

        assert count == 100

    def test_store_overlapping_features_deduplicates(self, repo, mock_s3_client, sample_features):
        """Test that overlapping features are deduplicated."""
        # Mock metadata and existing data
        existing_metadata = {
            "partitions": [{
                "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/features_20240101_000000_081500.parquet",
                "min_timestamp": "2024-01-01 00:00:00",
                "max_timestamp": "2024-01-01 08:15:00",
                "record_count": 100
            }]
        }

        # Mock existing parquet data
        existing_parquet_buffer = BytesIO()
        sample_features.to_parquet(existing_parquet_buffer, index=False)

        def mock_get_object(Bucket, Key):
            if "_metadata.json" in Key:
                return {"Body": BytesIO(json.dumps(existing_metadata).encode('utf-8'))}
            elif ".parquet" in Key:
                # Reset buffer position for each read
                existing_parquet_buffer.seek(0)
                return {"Body": existing_parquet_buffer}
            # Raise proper ClientError for other keys
            from botocore.exceptions import ClientError
            error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
            raise ClientError(error_response, 'GetObject')

        mock_s3_client.get_object.side_effect = mock_get_object

        # Store same timestamps again
        overlapping_features = sample_features.copy()
        overlapping_features["rsi_14"] = range(200, 300)

        count = repo.store_features_incrementally(overlapping_features, "BTCUSDT")

        # Should deduplicate - no new timestamps
        assert count == 0

    def test_metadata_is_created_on_s3(self, repo, mock_s3_client, sample_features):
        """Test that metadata file is created and uploaded to S3."""
        repo.store_features_incrementally(sample_features, "BTCUSDT")

        # Find metadata put_object calls
        metadata_calls = [call for call in mock_s3_client.put_object.call_args_list
                          if "_metadata.json" in call[1]['Key']]

        assert len(metadata_calls) > 0

        # Verify metadata structure
        metadata_call = metadata_calls[-1]  # Last call
        body = metadata_call[1]['Body']
        metadata = json.loads(body)

        assert "partitions" in metadata
        assert len(metadata["partitions"]) > 0
        assert all("s3_key" in p for p in metadata["partitions"])
        assert all("min_timestamp" in p for p in metadata["partitions"])
        assert all("max_timestamp" in p for p in metadata["partitions"])

    def test_empty_dataframe_returns_zero(self, repo):
        """Test that empty DataFrame returns 0 count."""
        empty_df = DataFrame(columns=["event_timestamp", "symbol", "rsi_14"])
        count = repo.store_features_incrementally(empty_df, "BTCUSDT")
        assert count == 0

    def test_missing_event_timestamp_raises_error(self, repo):
        """Test that missing event_timestamp column raises ValueError."""
        invalid_df = DataFrame({"symbol": ["BTCUSDT"], "rsi_14": [50]})

        with pytest.raises(ValueError, match="event_timestamp"):
            repo.store_features_incrementally(invalid_df, "BTCUSDT")


class TestStoreFeaturesBatch:
    """Test suite for batch storage operations."""

    def test_batch_store_initial_data(self, repo, mock_s3_client, sample_features):
        """Test batch storage of initial data."""
        result = repo.store_features_batch([
            {"symbol": "BTCUSDT", "features_df": sample_features}
        ])

        assert result["BTCUSDT"] == 100

        # Verify S3 put_object was called
        put_calls = [call for call in mock_s3_client.put_object.call_args_list
                     if call[1]['Key'].endswith('.parquet')]
        assert len(put_calls) > 0

    def test_batch_replaces_overlapping_partitions(self, repo, mock_s3_client, sample_features):
        """Test that batch mode replaces overlapping partitions."""
        # Mock existing metadata with overlapping data
        existing_metadata = {
            "partitions": [
                {
                    "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/features_20240101_000000_040000.parquet",
                    "min_timestamp": "2024-01-01 00:00:00",
                    "max_timestamp": "2024-01-01 04:00:00",
                    "record_count": 48
                },
                {
                    "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/features_20240101_040500_081500.parquet",
                    "min_timestamp": "2024-01-01 04:05:00",
                    "max_timestamp": "2024-01-01 08:15:00",
                    "record_count": 52
                }
            ]
        }

        # Mock existing parquet data (non-overlapping portion)
        non_overlapping_timestamps = pd.date_range("2024-01-01 00:00:00", periods=48, freq="5min")
        non_overlapping_df = DataFrame({
            "event_timestamp": non_overlapping_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(48),
            "ema_20": range(100, 148)
        })
        non_overlapping_df.index = pd.Index(range(len(non_overlapping_df)), dtype='int64')  # Avoid RangeIndex issues

        non_overlapping_buffer = BytesIO()
        non_overlapping_df.to_parquet(non_overlapping_buffer, index=False)

        def mock_get_object(Bucket, Key):
            if "_metadata.json" in Key:
                return {"Body": BytesIO(json.dumps(existing_metadata).encode('utf-8'))}
            elif "features_20240101_000000_040000.parquet" in Key:
                # Reset buffer for each read
                non_overlapping_buffer.seek(0)
                return {"Body": non_overlapping_buffer}
            # Raise proper ClientError for other keys
            from botocore.exceptions import ClientError
            error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
            raise ClientError(error_response, 'GetObject')

        mock_s3_client.get_object.side_effect = mock_get_object

        # Batch store overlapping data [04:00 - 12:15]
        replacement_timestamps = pd.date_range("2024-01-01 04:00:00", periods=100, freq="5min")
        replacement_features = DataFrame({
            "event_timestamp": replacement_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(500, 600),
            "ema_20": range(600, 700)
        })
        replacement_features.index = pd.Index(range(len(replacement_features)), dtype='int64')  # Avoid RangeIndex issues

        result = repo.store_features_batch([
            {"symbol": "BTCUSDT", "features_df": replacement_features}
        ])

        assert result["BTCUSDT"] == 100

        # Verify delete_object was called for overlapping partitions
        delete_calls = mock_s3_client.delete_object.call_args_list
        assert len(delete_calls) >= 1  # At least one overlapping partition deleted

    def test_batch_multiple_symbols(self, repo, mock_s3_client, sample_features):
        """Test batch storage for multiple symbols."""
        eth_features = sample_features.copy()
        eth_features["symbol"] = "ETHUSDT"

        result = repo.store_features_batch([
            {"symbol": "BTCUSDT", "features_df": sample_features},
            {"symbol": "ETHUSDT", "features_df": eth_features}
        ])

        assert result["BTCUSDT"] == 100
        assert result["ETHUSDT"] == 100

        # Verify put_object was called for both symbols
        put_calls = mock_s3_client.put_object.call_args_list
        btc_calls = [c for c in put_calls if "BTCUSDT" in c[1]['Key']]
        eth_calls = [c for c in put_calls if "ETHUSDT" in c[1]['Key']]

        assert len(btc_calls) > 0
        assert len(eth_calls) > 0


class TestPartitionOperations:
    """Test suite for partition-specific operations."""

    def test_find_overlapping_partitions(self, repo):
        """Test finding overlapping partitions."""
        partitions = [
            {"min_timestamp": "2024-01-01 00:00:00", "max_timestamp": "2024-01-01 04:00:00"},
            {"min_timestamp": "2024-01-01 08:00:00", "max_timestamp": "2024-01-01 12:00:00"},
            {"min_timestamp": "2024-01-01 16:00:00", "max_timestamp": "2024-01-01 20:00:00"}
        ]

        # Test overlap with first partition
        overlapping = repo._find_overlapping_partitions(
            partitions,
            pd.Timestamp("2024-01-01 02:00:00"),
            pd.Timestamp("2024-01-01 06:00:00")
        )
        assert len(overlapping) == 1
        assert overlapping[0]["min_timestamp"] == "2024-01-01 00:00:00"

        # Test overlap spanning multiple partitions
        overlapping = repo._find_overlapping_partitions(
            partitions,
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 18:00:00")
        )
        assert len(overlapping) == 2

        # Test no overlap
        overlapping = repo._find_overlapping_partitions(
            partitions,
            pd.Timestamp("2024-01-01 22:00:00"),
            pd.Timestamp("2024-01-01 23:00:00")
        )
        assert len(overlapping) == 0

    def test_load_partitions_from_s3(self, repo, mock_s3_client):
        """Test loading data from specific S3 partitions only."""
        # Create partition data
        partition1_timestamps = pd.date_range("2024-01-01 00:00:00", periods=50, freq="5min")
        partition1_df = DataFrame({
            "event_timestamp": partition1_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(50),
            "ema_20": range(100, 150)
        })
        partition1_df.index = pd.Index(range(len(partition1_df)), dtype='int64')  # Avoid RangeIndex issues

        partition1_buffer = BytesIO()
        partition1_df.to_parquet(partition1_buffer, index=False)

        partitions = [{
            "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/partition1.parquet",
            "min_timestamp": "2024-01-01 00:00:00",
            "max_timestamp": "2024-01-01 04:05:00",
            "record_count": 50
        }]

        # Mock get_object to return the partition data
        def mock_get_partition(Bucket, Key):
            partition1_buffer.seek(0)
            return {"Body": partition1_buffer}

        mock_s3_client.get_object.side_effect = mock_get_partition

        loaded = repo._load_partitions("BTCUSDT", partitions)

        assert not loaded.empty
        assert len(loaded) == 50

    def test_delete_partitions_from_s3(self, repo, mock_s3_client):
        """Test deleting specific partition files from S3."""
        partitions_to_delete = [
            {"s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/partition1.parquet"},
            {"s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/partition2.parquet"}
        ]

        repo._delete_partitions("BTCUSDT", partitions_to_delete)

        # Verify delete_object was called for each partition
        delete_calls = mock_s3_client.delete_object.call_args_list
        assert len(delete_calls) == 2

        # Verify correct keys were deleted
        deleted_keys = [call[1]['Key'] for call in delete_calls]
        assert "test-prefix/BTCUSDT/year=2024/month=01/day=01/partition1.parquet" in deleted_keys
        assert "test-prefix/BTCUSDT/year=2024/month=01/day=01/partition2.parquet" in deleted_keys

    def test_partition_loading_failure(self, repo, mock_s3_client) -> None:
        """Test partition loading failure."""
        # Mock get_object to fail
        mock_s3_client.get_object.side_effect = Exception("Load failed")

        partitions = [{"s3_key": "test-key"}]

        # Should log warning but return empty DataFrame
        result = repo._load_partitions("BTCUSDT", partitions)
        assert result.empty

    def test_partition_deletion_failure(self, repo, mock_s3_client) -> None:
        """Test partition deletion failure."""
        # Mock delete_object to fail
        mock_s3_client.delete_object.side_effect = Exception("Delete failed")

        partitions_to_delete = [{"s3_key": "test-key"}]

        # Should log error but not raise exception
        repo._delete_partitions("BTCUSDT", partitions_to_delete)

    def test_partition_deletion_empty_list(self, repo, mock_s3_client) -> None:
        """Test partition deletion with empty list."""
        # Should not call delete_object
        repo._delete_partitions("BTCUSDT", [])

        mock_s3_client.delete_object.assert_not_called()

    def test_partition_deletion_with_none_values(self, repo, mock_s3_client) -> None:
        """Test partition deletion with None values in list."""
        partitions_to_delete = [{"s3_key": "valid-key"}, None, {"s3_key": "another-key"}]

        # Should raise TypeError when trying to access None["s3_key"]
        with pytest.raises(TypeError):
            repo._delete_partitions("BTCUSDT", partitions_to_delete)


class TestMetadataOperations:
    """Test suite for metadata operations."""

    def test_metadata_uploaded_to_s3(self, repo, mock_s3_client, sample_features):
        """Test that metadata is correctly uploaded to S3."""
        repo.store_features_incrementally(sample_features, "BTCUSDT")

        # Find metadata put_object calls
        metadata_calls = [call for call in mock_s3_client.put_object.call_args_list
                          if "_metadata.json" in call[1]['Key']]

        assert len(metadata_calls) > 0

        # Verify metadata structure
        metadata_call = metadata_calls[-1]
        body = metadata_call[1]['Body']
        metadata = json.loads(body)

        assert "partitions" in metadata
        for partition in metadata["partitions"]:
            assert "s3_key" in partition
            assert "min_timestamp" in partition
            assert "max_timestamp" in partition
            assert "record_count" in partition

    def test_metadata_updated_on_new_storage(self, repo, mock_s3_client, sample_features):
        """Test that metadata is updated when storing new features."""
        # First storage - creates initial metadata
        repo.store_features_incrementally(sample_features, "BTCUSDT")

        initial_metadata_calls = len([c for c in mock_s3_client.put_object.call_args_list
                                      if "_metadata.json" in c[1]['Key']])

        # Second storage - should update metadata
        new_timestamps = pd.date_range("2024-01-02 00:00:00", periods=50, freq="5min")
        new_features = DataFrame({
            "event_timestamp": new_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(100, 150),
            "ema_20": range(150, 200)
        })
        new_features.index = pd.Index(range(len(new_features)), dtype='int64')  # Avoid RangeIndex issues

        # Mock metadata load to return initial metadata
        initial_metadata = {
            "partitions": [{
                "s3_key": "test-prefix/BTCUSDT/year=2024/month=01/day=01/features.parquet",
                "min_timestamp": "2024-01-01 00:00:00",
                "max_timestamp": "2024-01-01 08:15:00",
                "record_count": 100
            }]
        }

        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(json.dumps(initial_metadata).encode('utf-8'))
        }

        repo.store_features_incrementally(new_features, "BTCUSDT")

        updated_metadata_calls = len([c for c in mock_s3_client.put_object.call_args_list
                                       if "_metadata.json" in c[1]['Key']])

        # Should have more metadata calls (one for each storage)
        assert updated_metadata_calls > initial_metadata_calls

    def test_metadata_load_non_nosuchkey_error(self, repo, mock_s3_client, sample_features) -> None:
        """Test metadata loading with non-NoSuchKey error."""
        # Mock get_object to raise a different error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        # Should log warning and return None
        metadata = repo._load_metadata("BTCUSDT")
        assert metadata is None

    def test_metadata_save_failure(self, repo, mock_s3_client) -> None:
        """Test metadata save failure raises S3StorageException."""
        # Mock put_object to fail
        mock_s3_client.put_object.side_effect = Exception("Save failed")

        metadata = {"partitions": []}

        with pytest.raises(S3StorageException, match="Failed to save metadata"):
            repo._save_metadata("BTCUSDT", metadata)


class TestS3ErrorHandling:
    """Test suite for S3-specific error handling."""

    def test_handle_s3_connection_error(self, repo, mock_s3_client, sample_features) -> None:
        """Test handling of S3 connection errors."""
        from botocore.exceptions import BotoCoreError
        from drl_trading_adapter.adapter.feature_store.offline.parquet import S3StorageException

        # Mock put_object to fail when trying to store parquet
        mock_s3_client.put_object.side_effect = BotoCoreError()

        # Should wrap the error in S3StorageException
        with pytest.raises(S3StorageException):
            repo.store_features_incrementally(sample_features, "BTCUSDT")

    def test_handle_missing_metadata_gracefully(self, repo, mock_s3_client, sample_features) -> None:
        """Test that missing metadata is handled gracefully."""
        from botocore.exceptions import ClientError

        # Simulate NoSuchKey error for metadata
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}},
            "GetObject"
        )

        # Should treat as empty metadata and succeed
        count = repo.store_features_incrementally(sample_features, "BTCUSDT")
        assert count == 100

    def test_handle_malformed_metadata(self, repo, mock_s3_client, sample_features) -> None:
        """Test handling of malformed metadata in S3."""
        # Return invalid JSON
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(b"{ invalid json }")
        }

        # Should handle gracefully and treat as no metadata
        count = repo.store_features_incrementally(sample_features, "BTCUSDT")
        assert count == 100


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_store_duplicate_timestamps_within_batch(self, repo, sample_features) -> None:
        """Test storing features with duplicate timestamps within the same batch."""
        duplicate_timestamps = pd.date_range("2024-01-01", periods=10, freq="1h").tolist()
        duplicate_timestamps.extend(duplicate_timestamps[:5])  # Add 5 duplicates

        features = DataFrame({
            "event_timestamp": duplicate_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(15)
        })
        features.index = pd.Index(range(len(features)), dtype='int64')  # Avoid RangeIndex issues

        count = repo.store_features_incrementally(features, "BTCUSDT")

        # Should deduplicate within batch
        assert count == 10

    def test_concurrent_symbol_storage(self, repo, mock_s3_client, sample_features) -> None:
        """Test storing features for different symbols doesn't interfere."""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        for symbol in symbols:
            features = sample_features.copy()
            features["symbol"] = symbol
            repo.store_features_incrementally(features, symbol)

        # Verify put_object was called for each symbol
        put_calls = mock_s3_client.put_object.call_args_list

        for symbol in symbols:
            symbol_calls = [c for c in put_calls if symbol in c[1]['Key']]
            assert len(symbol_calls) > 0

    def test_large_batch_storage(self, repo, mock_s3_client) -> None:
        """Test storing a large batch of features."""
        # Create large dataset (10,000 records)
        timestamps = pd.date_range("2024-01-01", periods=10000, freq="1min")
        large_features = DataFrame({
            "event_timestamp": timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(10000),
            "ema_20": range(10000, 20000)
        })
        large_features.index = pd.Index(range(len(large_features)), dtype='int64')  # Avoid RangeIndex issues

        count = repo.store_features_incrementally(large_features, "BTCUSDT")

        assert count == 10000

        # Verify S3 operations were performed
        put_calls = [c for c in mock_s3_client.put_object.call_args_list
                     if c[1]['Key'].endswith('.parquet')]
        assert len(put_calls) > 0


class TestHelperMethods:
    """Test suite for helper methods that are not currently used."""

    def test_list_s3_objects(self, repo, mock_s3_client) -> None:
        """Test listing S3 objects with prefix."""
        # Mock list_objects_v2 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test-prefix/BTCUSDT/year=2024/month=01/day=01/file1.parquet'},
                {'Key': 'test-prefix/BTCUSDT/year=2024/month=01/day=01/file2.parquet'}
            ]
        }

        objects = repo._list_s3_objects("test-prefix/BTCUSDT/")

        assert len(objects) == 2
        assert 'file1.parquet' in objects[0]
        assert 'file2.parquet' in objects[1]

    def test_list_s3_objects_empty_bucket(self, repo, mock_s3_client) -> None:
        """Test listing S3 objects when bucket is empty."""
        mock_s3_client.list_objects_v2.return_value = {}

        objects = repo._list_s3_objects("test-prefix/BTCUSDT/")

        assert objects == []

    def test_list_s3_objects_error(self, repo, mock_s3_client) -> None:
        """Test listing S3 objects with error."""
        mock_s3_client.list_objects_v2.side_effect = Exception("List failed")

        with pytest.raises(S3StorageException, match="Failed to list S3 objects"):
            repo._list_s3_objects("test-prefix/BTCUSDT/")

    def test_load_parquet_from_s3(self, repo, mock_s3_client) -> None:
        """Test loading parquet file from S3."""
        # Create test parquet data
        test_data = DataFrame({
            'event_timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'symbol': 'BTCUSDT',
            'rsi_14': range(10)
        })
        test_data.index = pd.Index(range(len(test_data)), dtype='int64')  # Avoid RangeIndex issues

        buffer = BytesIO()
        test_data.to_parquet(buffer, index=False)
        buffer.seek(0)

        # Reset side_effect and set return_value
        mock_s3_client.get_object.side_effect = None
        mock_s3_client.get_object.return_value = {'Body': buffer}

        s3_key = "test-prefix/BTCUSDT/year=2024/month=01/day=01/test.parquet"
        loaded_data = repo._load_parquet_from_s3(s3_key)

        assert not loaded_data.empty
        assert len(loaded_data) == 10
        assert loaded_data['symbol'].iloc[0] == 'BTCUSDT'

    def test_load_parquet_from_s3_error(self, repo, mock_s3_client) -> None:
        """Test loading parquet file from S3 with error."""
        mock_s3_client.get_object.side_effect = Exception("Load failed")

        s3_key = "test-prefix/BTCUSDT/year=2024/month=01/day=01/test.parquet"

        with pytest.raises(S3StorageException, match="Failed to load parquet from s3://"):
            repo._load_parquet_from_s3(s3_key)
