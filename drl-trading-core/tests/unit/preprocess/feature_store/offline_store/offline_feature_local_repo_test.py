"""
Unit tests for OfflineFeatureLocalRepo.

Tests the local filesystem implementation of offline feature storage
with datetime-based organization and incremental storage capabilities.
"""

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo import (
    OfflineFeatureLocalRepo,
)


class TestOfflineFeatureLocalRepoInit:
    """Test class for OfflineFeatureLocalRepo initialization."""

    def test_init_with_valid_config(self, feature_store_config: FeatureStoreConfig) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        # Valid feature store configuration provided by fixture

        # When
        repo = OfflineFeatureLocalRepo(feature_store_config)

        # Then
        assert repo.config == feature_store_config
        assert repo.base_path == feature_store_config.offline_store_path


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
        empty_df = DataFrame(columns=["event_timestamp", "feature_1", "feature_2"])

        # When
        stored_count = offline_repo.store_features_incrementally(
            empty_df,
            eurusd_h1_symbol
        )

        # Then
        assert stored_count == 0
        assert not offline_repo.feature_exists(eurusd_h1_symbol)

    def test_store_features_incrementally_schema_validation_failure(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test schema validation when new features are missing required columns."""
        # Given
        # Store initial features with specific schema
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Create new features missing a required column
        incomplete_df = sample_features_df[["event_timestamp", "feature_1"]].copy()
        incomplete_df["event_timestamp"] = pd.Timestamp("2024-01-05 10:00:00")

        # When & Then
        with pytest.raises(ValueError, match="Schema validation failed.*missing columns"):
            offline_repo.store_features_incrementally(incomplete_df, eurusd_h1_symbol)


class TestOfflineFeatureLocalRepoLoadExisting:
    """Test class for loading existing features."""

    def test_load_existing_features_no_data(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when no data exists."""
        # Given
        # No existing features for the dataset

        # When
        result = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert result is None

    def test_load_existing_features_with_data(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading features when data exists."""
        # Given
        # Store some features first
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        result = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert result is not None
        assert len(result) == len(sample_features_df)
        assert "event_timestamp" in result.columns
        assert result["event_timestamp"].dtype == "datetime64[ns]"

    def test_load_existing_features_multiple_files(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test loading and combining features from multiple parquet files."""
        # Given
        # Store features from different days
        features_day1 = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00"), pd.Timestamp("2024-01-01 10:00:00")],
            "feature_1": [1.0, 2.0],
            "feature_2": [10.0, 20.0]
        })

        features_day2 = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-02 09:00:00"), pd.Timestamp("2024-01-02 10:00:00")],
            "feature_1": [3.0, 4.0],
            "feature_2": [30.0, 40.0]
        })

        offline_repo.store_features_incrementally(features_day1, eurusd_h1_symbol)
        offline_repo.store_features_incrementally(features_day2, eurusd_h1_symbol)

        # When
        result = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert result is not None
        assert len(result) == 4  # Combined from both days
        assert result["event_timestamp"].is_monotonic_increasing  # Should be sorted

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo.logger')
    def test_load_existing_features_corrupted_file(
        self,
        mock_logger: Mock,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test handling of corrupted parquet files during loading."""
        # Given
        # Store valid features first
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Create a corrupted file in the dataset path
        dataset_path = offline_repo._get_dataset_base_path(eurusd_h1_symbol)
        corrupted_file = os.path.join(dataset_path, "year=2024", "month=01", "day=01", "corrupted.parquet")
        os.makedirs(os.path.dirname(corrupted_file), exist_ok=True)
        with open(corrupted_file, "w") as f:
            f.write("corrupted content")

        # When
        result = offline_repo.load_existing_features(eurusd_h1_symbol)

        # Then
        assert result is not None  # Should still load valid files
        mock_logger.warning.assert_called()  # Should log warning about corrupted file


class TestOfflineFeatureLocalRepoUtilityMethods:
    """Test class for utility methods."""

    def test_feature_exists_false(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns False when no features exist."""
        # Given
        # No existing features

        # When
        exists = offline_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert exists is False

    def test_feature_exists_true(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test feature_exists returns True when features exist."""
        # Given
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        exists = offline_repo.feature_exists(eurusd_h1_symbol)

        # Then
        assert exists is True

    def test_get_feature_count_empty(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns 0 for empty dataset."""
        # Given
        # No existing features

        # When
        count = offline_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == 0

    def test_get_feature_count_with_data(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test get_feature_count returns correct count."""
        # Given
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # When
        count = offline_repo.get_feature_count(eurusd_h1_symbol)

        # Then
        assert count == len(sample_features_df)

    def test_get_dataset_base_path(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test dataset base path generation."""
        # Given
        # Repository with base path configured

        # When
        path = offline_repo._get_dataset_base_path(eurusd_h1_symbol)

        # Then
        expected_path = os.path.join(
            offline_repo.base_path,
            eurusd_h1_symbol
        )
        assert path == expected_path


class TestOfflineFeatureLocalRepoDatetimeOrganization:
    """Test class for datetime-based storage organization."""

    def test_datetime_organization_structure(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that files are organized in correct datetime structure."""
        # Given
        features_df = DataFrame({
            "event_timestamp": [
                pd.Timestamp("2024-01-15 09:00:00"),
                pd.Timestamp("2024-01-15 10:00:00"),
                pd.Timestamp("2024-02-01 09:00:00")
            ],
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0]
        })

        # When
        offline_repo.store_features_incrementally(features_df, eurusd_h1_symbol)

        # Then
        base_path = offline_repo._get_dataset_base_path(eurusd_h1_symbol)

        # Check that date-organized directories were created
        jan_path = os.path.join(base_path, "year=2024", "month=01", "day=15")
        feb_path = os.path.join(base_path, "year=2024", "month=02", "day=01")

        assert os.path.exists(jan_path)
        assert os.path.exists(feb_path)

        # Check that parquet files exist in the correct locations
        jan_files = [f for f in os.listdir(jan_path) if f.endswith('.parquet')]
        feb_files = [f for f in os.listdir(feb_path) if f.endswith('.parquet')]

        assert len(jan_files) == 1
        assert len(feb_files) == 1

    def test_filename_generation(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that filenames include timestamp information."""
        # Given
        features_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-15 09:30:45")],
            "feature_1": [1.0],
            "feature_2": [10.0]
        })

        # When
        offline_repo.store_features_incrementally(features_df, eurusd_h1_symbol)

        # Then
        base_path = offline_repo._get_dataset_base_path(eurusd_h1_symbol)
        date_path = os.path.join(base_path, "year=2024", "month=01", "day=15")

        files = [f for f in os.listdir(date_path) if f.endswith('.parquet')]
        assert len(files) == 1

        filename = files[0]
        assert filename.startswith("features_20240115_093045")
        assert filename.endswith(".parquet")


class TestOfflineFeatureLocalRepoSchemaValidation:
    """Test class for schema validation functionality."""

    def test_validate_schema_consistency_compatible_types(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test schema validation allows compatible numeric types."""
        # Given
        # Store initial features with int64 types
        initial_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_int": [1],  # int64
            "feature_float": [1.0]  # float64
        })
        offline_repo.store_features_incrementally(initial_df, eurusd_h1_symbol)

        # Create new features with compatible types
        new_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-02 09:00:00")],
            "feature_int": [2.0],  # float64 (compatible with int64)
            "feature_float": [2]   # int64 (compatible with float64)
        })

        # When & Then
        # Should not raise an exception
        offline_repo.store_features_incrementally(new_df, eurusd_h1_symbol)

    def test_validate_schema_consistency_new_columns_allowed(
        self,
        offline_repo: OfflineFeatureLocalRepo,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that new columns are allowed and logged."""
        # Given
        offline_repo.store_features_incrementally(sample_features_df, eurusd_h1_symbol)

        # Create features with additional column
        extended_df = sample_features_df.copy()
        extended_df["event_timestamp"] = pd.Timestamp("2024-01-05 09:00:00")
        extended_df["new_feature"] = [999.0] * len(extended_df)

        # When & Then
        # Should not raise an exception and should store successfully
        stored_count = offline_repo.store_features_incrementally(extended_df, eurusd_h1_symbol)
        assert stored_count == len(extended_df)

    @patch('drl_trading_core.preprocess.feature_store.offline_store.offline_feature_local_repo.logger')
    def test_validate_schema_consistency_type_mismatch_warning(
        self,
        mock_logger: Mock,
        offline_repo: OfflineFeatureLocalRepo,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that incompatible type changes generate warnings."""
        # Given
        initial_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_str": ["text"]  # string type
        })
        offline_repo.store_features_incrementally(initial_df, eurusd_h1_symbol)

        # Create features with incompatible type
        new_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-02 09:00:00")],
            "feature_str": [123]  # numeric type (incompatible)
        })

        # When
        offline_repo.store_features_incrementally(new_df, eurusd_h1_symbol)

        # Then
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Column type mismatch" in warning_call
