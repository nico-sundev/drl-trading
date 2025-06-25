"""
Unit tests for FeatureStoreSaveRepo.

Tests the feature store orchestration logic with mocked dependencies
to isolate the business logic from external infrastructure.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.feature_store_save_repo import (
    FeatureStoreSaveRepo,
)


class TestFeatureStoreSaveRepoInit:
    """Test class for FeatureStoreSaveRepo initialization."""

    def test_init_with_valid_dependencies(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock
    ) -> None:
        """Test successful initialization with valid dependencies."""
        # Given
        # Valid dependencies provided by fixtures

        # When
        repo = FeatureStoreSaveRepo(
            config=feature_store_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo
        )

        # Then
        assert repo.config == feature_store_config
        assert repo.feast_provider == mock_feast_provider
        assert repo.offline_repo == mock_offline_repo
        assert repo.feature_store == mock_feast_provider.get_feature_store.return_value
        mock_feast_provider.get_feature_store.assert_called_once()


class TestFeatureStoreSaveRepoOfflineStorage:
    """Test class for offline feature storage operations."""

    def test_store_computed_features_offline_success(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        sample_features_df: DataFrame,
        eurusd_h1_dataset_id: DatasetIdentifier,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test successful storage of computed features offline."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        # When
        feature_store_save_repo.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_dataset_id,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_called_once_with(
            sample_features_df,
            eurusd_h1_dataset_id
        )

    def test_store_computed_features_offline_empty_dataframe(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test handling of empty DataFrame during offline storage."""
        # Given
        empty_df = DataFrame()

        # When
        feature_store_save_repo.store_computed_features_offline(
            empty_df,
            eurusd_h1_dataset_id,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_not_called()

    def test_store_computed_features_offline_missing_timestamp_column(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier,
        feature_version_info: FeatureConfigVersionInfo
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
            feature_store_save_repo.store_computed_features_offline(
                invalid_df,
                eurusd_h1_dataset_id,
                feature_version_info
            )

    def test_store_computed_features_offline_no_new_features_stored(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        sample_features_df: DataFrame,
        eurusd_h1_dataset_id: DatasetIdentifier,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test handling when no new features are stored (duplicates)."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = 0

        # When
        feature_store_save_repo.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_dataset_id,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_called_once()
        # Should not create feature views when no new features are stored

    @patch('drl_trading_core.preprocess.feature_store.feature_store_save_repo.logger')
    def test_store_computed_features_offline_with_feature_views_creation(
        self,
        mock_logger: Mock,
        feature_store_save_repo: FeatureStoreSaveRepo,
        sample_features_df: DataFrame,
        eurusd_h1_dataset_id: DatasetIdentifier,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that feature views are created when new features are stored."""
        # Given
        stored_count = len(sample_features_df)
        mock_offline_repo.store_features_incrementally.return_value = stored_count

        mock_obs_fv = Mock()
        mock_reward_fv = Mock()
        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.create_feature_view.side_effect = [mock_obs_fv, mock_reward_fv]
        mock_feast_provider.create_feature_service.return_value = mock_feature_service

        # When
        feature_store_save_repo.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_dataset_id,
            feature_version_info
        )

        # Then
        # Verify feature views creation
        assert mock_feast_provider.create_feature_view.call_count == 2
        mock_feast_provider.create_feature_service.assert_called_once_with(
            feature_views=[mock_obs_fv, mock_reward_fv],
            dataset_id=eurusd_h1_dataset_id,
            feature_version_info=feature_version_info
        )

        # Verify feature store apply
        feature_store_save_repo.feature_store.apply.assert_called_once_with(
            [mock_obs_fv, mock_reward_fv, mock_feature_service]
        )

        # Verify logging
        mock_logger.info.assert_called()


class TestFeatureStoreSaveRepoBatchMaterialization:
    """Test class for batch materialization operations."""

    def test_batch_materialize_features_success(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        sample_features_df: DataFrame,
        eurusd_h1_dataset_id: DatasetIdentifier
    ) -> None:
        """Test successful batch materialization of features."""
        # Given
        expected_start_date = sample_features_df["event_timestamp"].min()
        expected_end_date = sample_features_df["event_timestamp"].max()

        # When
        feature_store_save_repo.batch_materialize_features(
            sample_features_df,
            eurusd_h1_dataset_id
        )

        # Then
        feature_store_save_repo.feature_store.materialize.assert_called_once_with(
            start_date=expected_start_date,
            end_date=expected_end_date
        )

    def test_batch_materialize_features_missing_timestamp_column(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier
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
            feature_store_save_repo.batch_materialize_features(
                invalid_df,
                eurusd_h1_dataset_id
            )


class TestFeatureStoreSaveRepoOnlineStore:
    """Test class for online store operations."""

    def test_push_features_to_online_store_success(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier
    ) -> None:
        """Test successful push of features to online store."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5],
            "feature_2": [10.0]
        })
        expected_feature_view_name = f"{eurusd_h1_dataset_id.symbol}_{eurusd_h1_dataset_id.timeframe.value}_features"

        # When
        feature_store_save_repo.push_features_to_online_store(
            single_record_df,
            eurusd_h1_dataset_id
        )

        # Then
        feature_store_save_repo.feature_store.write_to_online_store.assert_called_once_with(
            feature_view_name=expected_feature_view_name,
            df=single_record_df
        )

    def test_push_features_to_online_store_missing_timestamp_column(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier
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
            feature_store_save_repo.push_features_to_online_store(
                invalid_df,
                eurusd_h1_dataset_id
            )

    @patch('drl_trading_core.preprocess.feature_store.feature_store_save_repo.logger')
    def test_push_features_to_online_store_logging(
        self,
        mock_logger: Mock,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier
    ) -> None:
        """Test that online store push operations are logged at debug level."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })

        # When
        feature_store_save_repo.push_features_to_online_store(
            single_record_df,
            eurusd_h1_dataset_id
        )        # Then
        mock_logger.debug.assert_called_once()
        debug_message = mock_logger.debug.call_args[0][0]
        assert "Pushed 1 feature records to online store" in debug_message
        assert "EURUSD/1h" in debug_message


class TestFeatureStoreSaveRepoUtilityMethods:
    """Test class for utility methods."""

    def test_is_enabled_true(
        self,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock
    ) -> None:
        """Test is_enabled returns True when feature store is enabled."""
        # Given
        enabled_config = FeatureStoreConfig(
            enabled=True,
            repo_path="/tmp/test",
            offline_store_path="/tmp/offline",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=True,
            service_name="test_service",
            service_version="1.0.0"
        )

        # When
        repo = FeatureStoreSaveRepo(
            config=enabled_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo
        )

        # Then
        assert repo.is_enabled() is True

    def test_is_enabled_false(
        self,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock
    ) -> None:
        """Test is_enabled returns False when feature store is disabled."""
        # Given
        disabled_config = FeatureStoreConfig(
            enabled=False,
            repo_path="/tmp/test",
            offline_store_path="/tmp/offline",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        # When
        repo = FeatureStoreSaveRepo(
            config=disabled_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo
        )

        # Then
        assert repo.is_enabled() is False


class TestFeatureStoreSaveRepoPrivateMethods:
    """Test class for private method functionality."""

    def test_create_and_apply_feature_views_creation_sequence(
        self,
        feature_store_save_repo: FeatureStoreSaveRepo,
        eurusd_h1_dataset_id: DatasetIdentifier,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test the correct sequence of feature view creation."""
        # Given
        mock_obs_fv = Mock()
        mock_reward_fv = Mock()
        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.create_feature_view.side_effect = [mock_obs_fv, mock_reward_fv]
        mock_feast_provider.create_feature_service.return_value = mock_feature_service

        # When
        feature_store_save_repo._create_and_apply_feature_views(
            eurusd_h1_dataset_id,
            feature_version_info
        )

        # Then        # Verify observation space feature view creation
        obs_call = mock_feast_provider.create_feature_view.call_args_list[0]
        assert obs_call[1]["dataset_id"] == eurusd_h1_dataset_id
        assert obs_call[1]["feature_view_name"] == "observation_space_features"
        assert obs_call[1]["feature_role"] == FeatureRoleEnum.OBSERVATION_SPACE

        # Verify reward space feature view creation
        reward_call = mock_feast_provider.create_feature_view.call_args_list[1]
        assert reward_call[1]["dataset_id"] == eurusd_h1_dataset_id
        assert reward_call[1]["feature_view_name"] == "reward_space_features"
        assert reward_call[1]["feature_role"] == FeatureRoleEnum.REWARD_ENGINEERING

        # Verify feature service creation
        mock_feast_provider.create_feature_service.assert_called_once_with(
            feature_views=[mock_obs_fv, mock_reward_fv],
            dataset_id=eurusd_h1_dataset_id,
            feature_version_info=feature_version_info
        )

        # Verify Feast registry apply
        feature_store_save_repo.feature_store.apply.assert_called_once_with(
            [mock_obs_fv, mock_reward_fv, mock_feature_service]
        )
