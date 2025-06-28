"""
Unit tests for FeatureStoreSaveRepository.

Tests the feature store orchestration logic with mocked dependencies
to isolate the business logic from external infrastructure.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
    FeatureStoreSaveRepository,
)


class TestFeatureStoreSaveRepositoryInit:
    """Test class for FeatureStoreSaveRepository initialization."""

    def test_init_with_valid_dependencies(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test successful initialization with valid dependencies."""
        # Given
        # Valid dependencies provided by fixtures

        # When
        repo = FeatureStoreSaveRepository(
            config=feature_store_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo,
            feature_view_name_mapper=mock_feature_view_name_mapper
        )

        # Then
        assert repo.config == feature_store_config
        assert repo.feast_provider == mock_feast_provider
        assert repo.offline_repo == mock_offline_repo
        assert repo.feature_view_name_mapper == mock_feature_view_name_mapper
        assert repo.feature_store == mock_feast_provider.get_feature_store.return_value
        mock_feast_provider.get_feature_store.assert_called_once()


class TestFeatureStoreSaveRepositoryOfflineStorage:
    """Test class for offline feature storage operations."""

    def test_store_computed_features_offline_success(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test successful storage of computed features offline."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_called_once_with(
            sample_features_df,
            eurusd_h1_symbol
        )

    def test_store_computed_features_offline_empty_dataframe(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test handling of empty DataFrame during offline storage."""
        # Given
        empty_df = DataFrame()

        # When
        feature_store_save_repository.store_computed_features_offline(
            empty_df,
            eurusd_h1_symbol,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_not_called()

    def test_store_computed_features_offline_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0]
            # Missing event_timestamp column
        })

        # When & Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            feature_store_save_repository.store_computed_features_offline(
                invalid_df,
                eurusd_h1_symbol,
                feature_version_info
            )

    def test_store_computed_features_offline_no_new_features_stored(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test handling when no new features are stored (duplicates)."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = 0

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_version_info
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_called_once()
        # Should not create feature views when no new features are stored

    @patch('drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo.logger')
    def test_store_computed_features_offline_with_feature_views_creation(
        self,
        mock_logger: Mock,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
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
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_version_info
        )

        # Then
        # Verify feature views creation
        assert mock_feast_provider.create_feature_view.call_count == 2
        mock_feast_provider.create_feature_service.assert_called_once_with(
            feature_views=[mock_obs_fv, mock_reward_fv],
            symbol=eurusd_h1_symbol,
            feature_version_info=feature_version_info
        )

        # Verify feature store apply
        feature_store_save_repository.feature_store.apply.assert_called_once_with(
            [mock_obs_fv, mock_reward_fv, mock_feature_service]
        )

        # Verify logging
        mock_logger.info.assert_called()


class TestFeatureStoreSaveRepositoryBatchMaterialization:
    """Test class for batch materialization operations."""

    def test_batch_materialize_features_success(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test successful batch materialization of features."""
        # Given
        # Sample features DataFrame with event_timestamp

        # When
        feature_store_save_repository.batch_materialize_features(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Then
        feature_store_save_repository.feature_store.materialize.assert_called_once_with(
            start_date=sample_features_df["event_timestamp"].min(),
            end_date=sample_features_df["event_timestamp"].max()
        )

    def test_batch_materialize_features_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0]
            # Missing event_timestamp column
        })

        # When & Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            feature_store_save_repository.batch_materialize_features(
                invalid_df,
                eurusd_h1_symbol
            )

    @patch('drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo.logger')
    def test_batch_materialize_features_logging(
        self,
        mock_logger: Mock,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test that batch materialization is properly logged."""
        # Given
        # Sample features DataFrame

        # When
        feature_store_save_repository.batch_materialize_features(
            sample_features_df,
            eurusd_h1_symbol
        )

        # Then
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Materialized features for online serving" in log_message
        assert "EURUSD" in log_message


class TestFeatureStoreSaveRepositoryOnlineStore:
    """Test class for online store operations."""

    def test_push_features_to_online_store_success(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test successful push of features to online store."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        expected_feature_view_name = "observation_space_feature_view"
        mock_feature_view_name_mapper.map.return_value = expected_feature_view_name

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )

        # Then
        mock_feature_view_name_mapper.map.assert_called_once_with(feature_role)
        feature_store_save_repository.feature_store.write_to_online_store.assert_called_once_with(
            feature_view_name=expected_feature_view_name,
            df=single_record_df
        )

    def test_push_features_to_online_store_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.5]
            # Missing event_timestamp column
        })
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE

        # When & Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            feature_store_save_repository.push_features_to_online_store(
                invalid_df,
                eurusd_h1_symbol,
                feature_role
            )

    def test_push_features_to_online_store_unknown_feature_role(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test error handling when feature view name mapper fails."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })
        feature_role = Mock()  # Unknown feature role
        mock_feature_view_name_mapper.map.side_effect = ValueError("Unknown feature role")

        # When & Then
        with pytest.raises(ValueError, match="Unknown feature role"):
            feature_store_save_repository.push_features_to_online_store(
                single_record_df,
                eurusd_h1_symbol,
                feature_role
            )

    @patch('drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo.logger')
    def test_push_features_to_online_store_logging(
        self,
        mock_logger: Mock,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test that online store push operations are logged at debug level."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        mock_feature_view_name_mapper.map.return_value = "test_view"

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )

        # Then
        mock_logger.debug.assert_called_once()
        debug_message = mock_logger.debug.call_args[0][0]
        assert "Pushed 1 feature records" in debug_message
        assert "EURUSD" in debug_message


class TestFeatureStoreSaveRepositoryUtilityMethods:
    """Test class for utility methods."""

    def test_is_enabled_true(
        self,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock,
        mock_feature_view_name_mapper: Mock
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
        repo = FeatureStoreSaveRepository(
            config=enabled_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo,
            feature_view_name_mapper=mock_feature_view_name_mapper
        )

        # Then
        assert repo.is_enabled() is True

    def test_is_enabled_false(
        self,
        mock_feast_provider: Mock,
        mock_offline_repo: Mock,
        mock_feature_view_name_mapper: Mock
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
        repo = FeatureStoreSaveRepository(
            config=disabled_config,
            feast_provider=mock_feast_provider,
            offline_repo=mock_offline_repo,
            feature_view_name_mapper=mock_feature_view_name_mapper
        )

        # Then
        assert repo.is_enabled() is False


class TestFeatureStoreSaveRepositoryPrivateMethods:
    """Test class for private methods (through public interface)."""

    def test_create_and_apply_feature_views_called_correctly(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that _create_and_apply_feature_views is called with correct parameters."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        mock_obs_fv = Mock()
        mock_reward_fv = Mock()
        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.create_feature_view.side_effect = [mock_obs_fv, mock_reward_fv]
        mock_feast_provider.create_feature_service.return_value = mock_feature_service

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_version_info
        )

        # Then
        # Verify observation space feature view creation
        observation_call_args = mock_feast_provider.create_feature_view.call_args_list[0]
        assert observation_call_args[1]["symbol"] == eurusd_h1_symbol
        assert observation_call_args[1]["feature_view_name"] == "observation_space_feature_view"
        assert observation_call_args[1]["feature_role"] == FeatureRoleEnum.OBSERVATION_SPACE
        assert observation_call_args[1]["feature_version_info"] == feature_version_info

        # Verify reward engineering feature view creation
        reward_call_args = mock_feast_provider.create_feature_view.call_args_list[1]
        assert reward_call_args[1]["symbol"] == eurusd_h1_symbol
        assert reward_call_args[1]["feature_view_name"] == "reward_engineering_feature_view"
        assert reward_call_args[1]["feature_role"] == FeatureRoleEnum.REWARD_ENGINEERING
        assert reward_call_args[1]["feature_version_info"] == feature_version_info


class TestFeatureStoreSaveRepositoryErrorHandling:
    """Test class for error handling scenarios."""

    def test_store_computed_features_offline_feast_provider_exception(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test error handling when feast provider raises exception."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)
        mock_feast_provider.create_feature_view.side_effect = Exception("Feast provider error")

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            feature_store_save_repository.store_computed_features_offline(
                sample_features_df,
                eurusd_h1_symbol,
                feature_version_info
            )

    def test_batch_materialize_features_feast_store_exception(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when feast store materialization fails."""
        # Given
        feature_store_save_repository.feature_store.materialize.side_effect = Exception("Materialization failed")

        # When & Then
        with pytest.raises(Exception, match="Materialization failed"):
            feature_store_save_repository.batch_materialize_features(
                sample_features_df,
                eurusd_h1_symbol
            )

    def test_push_features_to_online_store_feast_store_exception(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test error handling when online store write fails."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })
        feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        mock_feature_view_name_mapper.map.return_value = "test_view"
        feature_store_save_repository.feature_store.write_to_online_store.side_effect = Exception("Online store error")

        # When & Then
        with pytest.raises(Exception, match="Online store error"):
            feature_store_save_repository.push_features_to_online_store(
                single_record_df,
                eurusd_h1_symbol,
                feature_role
            )


@pytest.mark.parametrize("feature_role,expected_view_name", [
    (FeatureRoleEnum.OBSERVATION_SPACE, "observation_space_feature_view"),
    (FeatureRoleEnum.REWARD_ENGINEERING, "reward_engineering_feature_view"),
])
class TestFeatureStoreSaveRepositoryParametrized:
    """Parametrized tests for different feature roles."""

    def test_push_features_to_online_store_different_roles(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_feature_view_name_mapper: Mock,
        feature_role: FeatureRoleEnum,
        expected_view_name: str
    ) -> None:
        """Test push to online store with different feature roles."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "feature_1": [1.5]
        })
        mock_feature_view_name_mapper.map.return_value = expected_view_name

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )

        # Then
        mock_feature_view_name_mapper.map.assert_called_once_with(feature_role)
        feature_store_save_repository.feature_store.write_to_online_store.assert_called_once_with(
            feature_view_name=expected_view_name,
            df=single_record_df
        )
