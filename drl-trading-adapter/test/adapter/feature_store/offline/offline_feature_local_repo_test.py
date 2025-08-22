"""
Unit tests for OfflineFeatureLocalRepo.

Tests the local filesystem implementation of offline feature storage
with datetime-based organization and incremental storage capabilities.
"""

import os
from unittest.mock import Mock, patch

from drl_trading_adapter.adapter.feature_store.offline.offline_feature_local_repo import OfflineFeatureLocalRepo
import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from pandas import DataFrame


class TestOfflineFeatureLocalRepoInit:
    """Test class for OfflineFeatureLocalRepo initialization."""

    def test_init_with_valid_config(self, feature_store_config: FeatureStoreConfig) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        # Valid feature store configuration provided by fixture

        # When
        repo = OfflineFeatureLocalRepo(feature_store_config)

        # Then
        assert repo.base_path == feature_store_config.local_repo_config.repo_path


class TestOfflineFeatureLocalRepoStoreIncremental:
    """Test class for incremental feature storage operations."""

    def test_store_features_incrementally_new_dataset(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features for a new dataset with no existing data."""
        # Given
        # Fresh dataset with no existing features
        assert not offline_repo.feature_exists(eurusd_h1_symbol)

        # When
        stored_count = offline_repo.store_features_incrementally(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == len(sample_features_df)
        assert offline_repo.feature_exists(eurusd_h1_symbol)
        assert offline_repo.get_feature_count(eurusd_h1_symbol) == len(sample_features_df)

    def test_store_features_incrementally_with_duplicates(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test storing features when some timestamps already exist."""
        # Given
        # Store initial features
        initial_count = offline_repo.store_features_incrementally(
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
        stored_count = offline_repo.store_features_incrementally(
            overlapping_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == 1  # Only the new row should be stored
        assert offline_repo.get_feature_count(eurusd_h1_symbol) == initial_count + 1

    def test_store_features_incrementally_missing_timestamp_column(
        self,
        offline_repo: OfflineFeatureLocalRepo,
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
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            offline_repo.store_features_incrementally(invalid_df, eurusd_h1_symbol)

    def test_store_features_incrementally_empty_dataframe(
        self,
        offline_repo: OfflineFeatureLocalRepo,
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
        assert not offline_repo.feature_exists(eurusd_h1_symbol)


class TestOfflineFeatureLocalRepoLoadExisting:
    """Test class for loading existing features."""

    def test_load_existing_features_no_data(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when no data exists."""
        # Given
        # No existing features for symbol
        assert not offline_repo.feature_exists(eurusd_h1_symbol)

        # When
        loaded_features = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert loaded_features is None

    def test_load_existing_features_with_data(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when data exists."""
        # Given
        # Store features first
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        loaded_features = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert loaded_features is not None
        assert len(loaded_features) == len(sample_features_df)
        pd.testing.assert_frame_equal(
            loaded_features.sort_values("event_timestamp").reset_index(drop=True),
            sample_features_df.sort_values("event_timestamp").reset_index(drop=True)
        )


class TestOfflineFeatureLocalRepoUtilityMethods:
    """Test class for utility methods."""

    def test_feature_exists_false_for_new_symbol(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns False for new symbol."""
        # Given
        # Fresh symbol with no stored features

        # When
        exists = offline_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert not exists

    def test_feature_exists_true_after_storage(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns True after storing features."""
        # Given
        # Store features
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        exists = offline_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert exists

    def test_get_feature_count_zero_for_new_symbol(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns 0 for new symbol."""
        # Given
        # Fresh symbol with no stored features

        # When
        count = offline_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == 0

    def test_get_feature_count_correct_after_storage(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns correct count after storage."""
        # Given
        # Store features
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        count = offline_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == len(sample_features_df)


class TestOfflineFeatureLocalRepoDatetimeOrganization:
    """Test class for datetime-based storage organization."""

    def test_datetime_organization_structure(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that features are organized by datetime structure."""
        # Given
        # Store features with known timestamps
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        # Check the generated directory structure
        base_path = offline_repo._get_dataset_base_path(eurusd_h1_symbol)

        # Then
        assert os.path.exists(base_path)
        # Should have datetime-organized subdirectories
        subdirs = []
        for root, _dirs, files in os.walk(base_path):
            if files:  # Directories with parquet files
                subdirs.append(root)

        assert len(subdirs) > 0, "Should have created datetime-organized directories"

    @patch('os.walk')
    def test_load_from_multiple_partitions(
        self,
        mock_walk: Mock,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features from multiple datetime partitions."""
        # Given
        mock_walk.return_value = [
            ("/test/EURUSD/year=2024/month=01/day=01", [], ["features_part1.parquet"]),
            ("/test/EURUSD/year=2024/month=01/day=02", [], ["features_part2.parquet"])
        ]

        with patch('pandas.read_parquet') as mock_read:
            # Mock parquet files content
            df1 = DataFrame({
                "event_timestamp": [pd.Timestamp("2024-01-01 10:00:00")],
                "feature_1": [1.0]
            })
            df2 = DataFrame({
                "event_timestamp": [pd.Timestamp("2024-01-02 10:00:00")],
                "feature_1": [2.0]
            })
            mock_read.side_effect = [df1, df2]

            # When
            result = offline_repo.load_existing_features(eurusd_h1_symbol)

            # Then
            assert result is not None
            assert len(result) == 2
            assert len(mock_read.call_args_list) == 2


class TestOfflineFeatureLocalRepoSchemaValidation:
    """Test class for schema validation between existing and new features."""

    def test_schema_consistency_validation_success(
        self,
        offline_repo: OfflineFeatureLocalRepo,
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

    def test_schema_consistency_validation_failure(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test schema validation failure with inconsistent schemas."""
        # Given
        # Store initial features
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Create new features with different schema (missing column)
        incompatible_features = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-05 10:00:00")],
            "different_feature": [999.0]  # Different column name
        })

        # When & Then
        with pytest.raises(ValueError, match="Schema mismatch"):
            offline_repo.store_features_incrementally(incompatible_features, eurusd_h1_symbol)
