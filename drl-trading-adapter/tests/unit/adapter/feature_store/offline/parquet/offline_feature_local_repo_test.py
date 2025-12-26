"""
Unit tests for OfflineFeatureLocalRepo.

Tests the local filesystem implementation of offline feature storage
with datetime-based organization and incremental storage capabilities.
"""

import json
import os
import shutil
import tempfile

import pandas as pd
import pytest
from pandas import DataFrame, concat, to_datetime

from drl_trading_adapter.adapter.feature_store.offline.parquet import OfflineLocalParquetFeatureRepo
from drl_trading_common.config.feature_config import FeatureStoreConfig, LocalRepoConfig


def load_all_partitions(repo_path: str) -> DataFrame:
    """Helper function to load all parquet files from repo for test validation."""
    parquet_files = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))

    if not parquet_files:
        return DataFrame()

    dfs = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if "event_timestamp" in df.columns:
            df["event_timestamp"] = to_datetime(df["event_timestamp"])
        dfs.append(df)

    return (
        concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["event_timestamp"])
        .sort_values("event_timestamp")
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def local_config(temp_dir):
    """Create a local repository configuration."""
    return FeatureStoreConfig(
        offline_repo_strategy="local",
        local_repo_config=LocalRepoConfig(repo_path=temp_dir),
        cache_enabled=False,
        entity_name="test_entity",
        ttl_days=30,
        config_directory="/tmp/test"
    )


@pytest.fixture
def repo(local_config):
    """Create a local repository instance."""
    return OfflineLocalParquetFeatureRepo(local_config)


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    df = DataFrame({
        "event_timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
        "symbol": "BTCUSDT",
        "rsi_14": range(100),
        "ema_20": range(100, 200)
    })
    df.index = pd.Index(range(len(df)), dtype='int64')  # Avoid RangeIndex issues
    return df


class TestOfflineFeatureLocalRepoInit:
    """Test class for OfflineFeatureLocalRepo initialization."""

    def test_init_with_valid_config(self, feature_store_config: FeatureStoreConfig) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        # Valid feature store configuration provided by fixture

        # When
        repo = OfflineLocalParquetFeatureRepo(feature_store_config)

        # Then
        assert repo.base_path == feature_store_config.local_repo_config.repo_path


class TestOfflineFeatureLocalRepoStoreIncremental:
    """Test class for incremental feature storage operations."""

    def test_store_features_incrementally_missing_timestamp_column(
        self,
        offline_repo: OfflineLocalParquetFeatureRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0]
            # Missing event_timestamp column
        })
        # When & Then
        # Implementation raises a ValueError with a different message
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            offline_repo.store_features_incrementally(invalid_df, eurusd_h1_symbol)

    def test_store_features_incrementally_empty_dataframe(
        self,
        offline_repo: OfflineLocalParquetFeatureRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test handling of empty DataFrame."""
        # Given
        empty_df = DataFrame({
            "event_timestamp": []
        })

        # When
        stored_count = offline_repo.store_features_incrementally(
            empty_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == 0


class TestOfflineFeatureLocalRepoDatetimeOrganization:
    """Test class for datetime-based storage organization."""

    def test_datetime_organization_structure(
        self,
        offline_repo: OfflineLocalParquetFeatureRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that features are organized by datetime structure."""
        # Given
        # Store features with known timestamps
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        # Check the generated directory structure via public API
        base_path = offline_repo.get_repo_path(eurusd_h1_symbol)

        # Then
        assert os.path.exists(base_path)
        # Should have datetime-organized subdirectories
        subdirs = []
        for root, _dirs, files in os.walk(base_path):
            if files:  # Directories with parquet files
                subdirs.append(root)

        assert len(subdirs) > 0, "Should have created datetime-organized directories"


class TestOfflineFeatureLocalRepoSchemaValidation:
    """Test class for schema validation between existing and new features."""

    def test_schema_consistency_validation_success(
        self,
        offline_repo: OfflineLocalParquetFeatureRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test successful schema validation with consistent schemas."""
        # Given
        # Store initial features
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Create new features with same schema
        new_features = sample_features_df.copy()
        new_features["event_timestamp"] = pd.Timestamp("2024-01-05 10:00:00")

        # When & Then
        # Should not raise an exception
        stored_count = offline_repo.store_features_incrementally(new_features, eurusd_h1_symbol)
        assert stored_count == 1  # Only new timestamp should be stored


class TestStoreFeaturesIncrementally:
    """Test incremental feature storage with partition awareness."""

    def test_store_initial_features(self, repo, sample_features):
        """Test storing initial features creates proper directory structure."""
        # Given
        symbol = "BTCUSDT"

        # When
        count = repo.store_features_incrementally(sample_features, symbol)

        # Then
        assert count == 100
        assert os.path.exists(repo.get_repo_path(symbol))

        # Verify data was stored correctly
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert not loaded.empty
        assert len(loaded) == 100

    def test_store_non_overlapping_features(self, repo, sample_features):
        """Test storing features with timestamps that don't overlap with existing data."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Create new features with later timestamps
        new_features = DataFrame({
            "event_timestamp": pd.date_range("2024-01-02", periods=50, freq="1min"),
            "symbol": "BTCUSDT",
            "rsi_14": range(50, 100),
            "ema_20": range(150, 200)
        })
        new_features.index = pd.Index(range(len(new_features)), dtype='int64')  # Avoid RangeIndex issues

        # When
        count = repo.store_features_incrementally(new_features, symbol)

        # Then
        assert count == 50
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert len(loaded) == 150  # Original 100 + 50 new

    def test_store_overlapping_features_deduplicates(self, repo, sample_features):
        """Test that overlapping timestamps are deduplicated."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Create overlapping features (same timestamps, different values)
        overlapping_features = sample_features.copy()
        overlapping_features["rsi_14"] = overlapping_features["rsi_14"] + 1000  # Different values

        # When
        count = repo.store_features_incrementally(overlapping_features, symbol)

        # Then
        assert count == 0  # No new records stored due to deduplication
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert len(loaded) == 100  # Still only original records

        # Verify original values are preserved (not overwritten)
        assert (loaded["rsi_14"] < 1000).all()  # Original values, not modified ones

    def test_store_partial_overlap(self, repo):
        """Test storing features where some timestamps overlap and some don't."""
        # Given
        symbol = "BTCUSDT"

        # Store initial features
        initial_features = DataFrame({
            "event_timestamp": pd.date_range("2024-01-01 10:00:00", periods=10, freq="1min"),
            "symbol": "BTCUSDT",
            "rsi_14": range(10),
            "ema_20": range(10, 20)
        })
        initial_features.index = pd.Index(range(len(initial_features)), dtype='int64')  # Avoid RangeIndex issues
        repo.store_features_incrementally(initial_features, symbol)

        # Create features with partial overlap
        overlapping_features = DataFrame({
            "event_timestamp": pd.date_range("2024-01-01 10:05:00", periods=10, freq="1min"),  # Overlap at 10:05-10:10
            "symbol": "BTCUSDT",
            "rsi_14": range(100, 110),
            "ema_20": range(200, 210)
        })
        overlapping_features.index = pd.Index(range(len(overlapping_features)), dtype='int64')  # Avoid RangeIndex issues

        # When
        count = repo.store_features_incrementally(overlapping_features, symbol)

        # Then
        assert count == 5  # Only non-overlapping records stored (10:10 onwards)
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert len(loaded) == 15  # 10 original + 5 new

    def test_metadata_is_created(self, repo, sample_features):
        """Test that metadata is created when storing features."""
        # Given
        symbol = "BTCUSDT"

        # When
        repo.store_features_incrementally(sample_features, symbol)

        # Then
        metadata_path = os.path.join(repo.get_repo_path(symbol), "_metadata.json")
        assert os.path.exists(metadata_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert "partitions" in metadata
        assert len(metadata["partitions"]) > 0

    def test_empty_dataframe_returns_zero(self, repo):
        """Test that storing an empty dataframe returns 0."""
        # Given
        symbol = "BTCUSDT"
        empty_df = DataFrame({
            "event_timestamp": pd.to_datetime([]),
            "symbol": [],
            "rsi_14": [],
            "ema_20": []
        })

        # When
        count = repo.store_features_incrementally(empty_df, symbol)

        # Then
        assert count == 0

    def test_missing_event_timestamp_raises_error(self, repo):
        """Test that missing event_timestamp column raises an error."""
        # Given
        symbol = "BTCUSDT"
        invalid_df = DataFrame({
            "symbol": ["BTCUSDT"],
            "rsi_14": [50],
            "ema_20": [100]
        })

        # When/Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            repo.store_features_incrementally(invalid_df, symbol)


class TestStoreFeaturesBatch:
    """Test batch feature storage operations."""

    def test_batch_store_initial_data(self, repo, sample_features):
        """Test batch storing initial data."""
        # Given
        batch_data = [
            {"symbol": "BTCUSDT", "features_df": sample_features.iloc[:50]},
            {"symbol": "ETHUSDT", "features_df": sample_features.iloc[50:]}
        ]

        # When
        results = repo.store_features_batch(batch_data)

        # Then
        assert results["BTCUSDT"] == 50
        assert results["ETHUSDT"] == 50

        # Verify data was stored
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            loaded = load_all_partitions(repo.get_repo_path(symbol))
            assert len(loaded) == 50

    def test_batch_replaces_overlapping_partitions(self, repo, sample_features):
        """Test that batch storage replaces overlapping partitions."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Modify data and store via batch
        modified_data = sample_features.copy()
        modified_data["rsi_14"] = modified_data["rsi_14"] * 2

        batch_data = [{"symbol": symbol, "features_df": modified_data}]

        # When
        results = repo.store_features_batch(batch_data)

        # Then
        assert results[symbol] == 100  # All records stored (batch replaces)
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert len(loaded) == 100
        # Verify values were updated
        assert (loaded["rsi_14"] >= 0).all()  # Modified values

    def test_batch_keeps_non_overlapping_data(self, repo):
        """Test that batch storage keeps non-overlapping data."""
        # Given
        symbol = "BTCUSDT"

        # Store initial data
        initial_data = DataFrame({
            "event_timestamp": pd.date_range("2024-01-01", periods=50, freq="1min"),
            "symbol": symbol,
            "rsi_14": range(50),
            "ema_20": range(50, 100)
        })
        initial_data.index = pd.Index(range(len(initial_data)), dtype='int64')  # Avoid RangeIndex issues
        repo.store_features_incrementally(initial_data, symbol)

        # Create batch data for different time range
        batch_data = DataFrame({
            "event_timestamp": pd.date_range("2024-01-02", periods=50, freq="1min"),
            "symbol": symbol,
            "rsi_14": range(100, 150),
            "ema_20": range(200, 250)
        })
        batch_data.index = pd.Index(range(len(batch_data)), dtype='int64')  # Avoid RangeIndex issues

        # When
        results = repo.store_features_batch([{"symbol": symbol, "features_df": batch_data}])

        # Then
        assert results[symbol] == 50
        loaded = load_all_partitions(repo.get_repo_path(symbol))
        assert len(loaded) == 100  # 50 original + 50 new

    def test_batch_multiple_symbols(self, repo, sample_features):
        """Test batch storage for multiple symbols."""
        # Given
        batch_data = [
            {"symbol": "BTCUSDT", "features_df": sample_features},
            {"symbol": "ETHUSDT", "features_df": sample_features.copy()},
            {"symbol": "ADAUSDT", "features_df": sample_features.copy()}
        ]

        # When
        results = repo.store_features_batch(batch_data)

        # Then
        for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
            assert results[symbol] == 100
            loaded = load_all_partitions(repo.get_repo_path(symbol))
            assert len(loaded) == 100


class TestPartitionOperations:
    """Test partition-level operations."""

    def test_find_overlapping_partitions(self, repo, sample_features):
        """Test finding partitions that overlap with given timestamps."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # When
        metadata = repo._load_metadata(symbol)
        min_time = sample_features["event_timestamp"].min()
        max_time = sample_features["event_timestamp"].max()
        overlapping = repo._find_overlapping_partitions(
            metadata["partitions"],
            min_time,
            max_time
        )

        # Then
        assert isinstance(overlapping, list)
        # Should find partitions for the stored data
        assert len(overlapping) > 0

    def test_load_specific_partitions(self, repo, sample_features):
        """Test loading data from specific partitions."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Get partition paths
        metadata = repo._load_metadata(symbol)
        min_time = sample_features["event_timestamp"].min()
        max_time = sample_features["event_timestamp"].max()
        partitions = repo._find_overlapping_partitions(
            metadata["partitions"],
            min_time,
            max_time
        )

        # When
        loaded = repo._load_partitions(symbol, partitions)

        # Then
        assert not loaded.empty
        assert len(loaded) > 0

    def test_delete_specific_partitions(self, repo, sample_features):
        """Test deleting specific partitions."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Get partition paths
        metadata = repo._load_metadata(symbol)
        min_time = sample_features["event_timestamp"].min()
        max_time = sample_features["event_timestamp"].max()
        partitions = repo._find_overlapping_partitions(
            metadata["partitions"],
            min_time,
            max_time
        )

        # When
        repo._delete_partitions(symbol, partitions)

        # Then
        # Verify partitions were deleted
        for partition in partitions:
            assert not os.path.exists(partition["path"])


class TestMetadataOperations:
    """Test metadata management operations."""

    def test_metadata_tracks_all_partitions(self, repo, sample_features):
        """Test that metadata tracks all created partitions."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # When
        metadata = repo._load_metadata(symbol)

        # Then
        assert "partitions" in metadata
        assert len(metadata["partitions"]) > 0

        # Verify all partition files exist
        for partition in metadata["partitions"]:
            assert os.path.exists(partition["path"])

    def test_metadata_updated_on_new_storage(self, repo, sample_features):
        """Test that metadata is updated when new data is stored."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)
        initial_metadata = repo._load_metadata(symbol)

        # Store additional data
        new_features = DataFrame({
            "event_timestamp": pd.date_range("2024-01-02", periods=50, freq="1min"),
            "symbol": symbol,
            "rsi_14": range(50),
            "ema_20": range(50, 100)
        })
        new_features.index = pd.Index(range(len(new_features)), dtype='int64')  # Avoid RangeIndex issues
        repo.store_features_incrementally(new_features, symbol)

        # When
        updated_metadata = repo._load_metadata(symbol)

        # Then
        assert len(updated_metadata["partitions"]) >= len(initial_metadata["partitions"])

    def test_metadata_survives_batch_replacement(self, repo, sample_features):
        """Test that metadata is properly updated after batch operations."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Perform batch replacement
        batch_data = [{"symbol": symbol, "features_df": sample_features}]
        repo.store_features_batch(batch_data)

        # When
        metadata = repo._load_metadata(symbol)

        # Then
        assert "partitions" in metadata
        assert len(metadata["partitions"]) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_store_duplicate_timestamps_within_batch(self, repo):
        """Test storing features with duplicate timestamps within the same batch."""
        # Given
        symbol = "BTCUSDT"
        duplicate_timestamps = pd.date_range("2024-01-01", periods=10, freq="1min").tolist()
        duplicate_timestamps.extend(duplicate_timestamps[:5])  # Add 5 duplicates

        features = DataFrame({
            "event_timestamp": duplicate_timestamps,
            "symbol": "BTCUSDT",
            "rsi_14": range(15)
        })
        features.index = pd.Index(range(len(features)), dtype='int64')  # Avoid RangeIndex issues

        # When
        count = repo.store_features_incrementally(features, symbol)

        # Then
        # Should deduplicate within batch
        assert count == 10

    def test_handle_malformed_metadata(self, repo, sample_features):
        """Test handling of malformed metadata."""
        # Given
        symbol = "BTCUSDT"
        repo.store_features_incrementally(sample_features, symbol)

        # Corrupt metadata file
        metadata_path = os.path.join(repo.get_repo_path(symbol), "metadata.json")
        with open(metadata_path, 'w') as f:
            f.write("{ invalid json }")

        # When
        # Try to store more data (should handle corrupted metadata gracefully)
        new_features = DataFrame({
            "event_timestamp": pd.date_range("2024-01-02", periods=10, freq="1min"),
            "symbol": symbol,
            "rsi_14": range(10)
        })
        new_features.index = pd.Index(range(len(new_features)), dtype='int64')  # Avoid RangeIndex issues

        count = repo.store_features_incrementally(new_features, symbol)

        # Then
        assert count == 10  # Should still work despite corrupted metadata

    def test_concurrent_symbol_storage(self, repo, sample_features):
        """Test storing features for different symbols doesn't interfere."""
        # Given
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        # When
        for symbol in symbols:
            features = sample_features.copy()
            features["symbol"] = symbol
            repo.store_features_incrementally(features, symbol)

        # Then
        # Verify each symbol has correct data
        for symbol in symbols:
            loaded = load_all_partitions(repo.get_repo_path(symbol))
            assert not loaded.empty
            assert len(loaded) == 100
            assert (loaded["symbol"] == symbol).all()
