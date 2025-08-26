"""
Unit tests for FeatureStoreSaveRepository.

Tests the feature store orchestration logic with mocked dependencies
to isolate the business logic from external infrastructure.
"""

from unittest.mock import Mock, patch

from drl_trading_preprocess.adapter.feature_store import FeatureStoreSaveRepository
import pandas as pd
import pytest
from drl_trading_common.enum import FeatureRoleEnum
from drl_trading_core.common.model.feature_view_request import FeatureViewRequest
from pandas import DataFrame


class TestFeatureStoreSaveRepositoryInit:
    """Test class for FeatureStoreSaveRepository initialization."""

    def test_init_with_valid_dependencies(
        self,
        mock_feast_provider: Mock,
        mock_feature_view_name_mapper: Mock
    ) -> None:
        """Test successful initialization with valid dependencies."""
        # Given
        # Valid dependencies provided by fixtures

        # When
        repo = FeatureStoreSaveRepository(
            feast_provider=mock_feast_provider,
            feature_view_name_mapper=mock_feature_view_name_mapper,
        )

        # Then
        assert repo.feast_provider == mock_feast_provider
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
        feature_view_requests: list[FeatureViewRequest]
    ) -> None:
        """Test successful storage of computed features offline."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_view_requests
        )

        # Then
        # Implementation converts timestamps to UTC, so we need to check the actual call
        call_args = mock_offline_repo.store_features_incrementally.call_args
        actual_df, actual_symbol = call_args[0]

        # Verify the symbol is correct
        assert actual_symbol == eurusd_h1_symbol

        # Verify DataFrame structure (implementation may have added timezone info)
        assert len(actual_df) == len(sample_features_df)
        assert list(actual_df.columns) == list(sample_features_df.columns)

        # Verify core data values are preserved (ignoring potential timezone changes)
        pd.testing.assert_series_equal(
            actual_df["rsi_14"], sample_features_df["rsi_14"], check_names=True
        )
        pd.testing.assert_series_equal(
            actual_df["sma_20"], sample_features_df["sma_20"], check_names=True
        )

    def test_store_computed_features_offline_empty_dataframe(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_view_requests: list[FeatureViewRequest]
    ) -> None:
        """Test handling of empty DataFrame during offline storage."""
        # Given
        empty_df = DataFrame()

        # When
        feature_store_save_repository.store_computed_features_offline(
            empty_df,
            eurusd_h1_symbol,
            feature_view_requests
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_not_called()

    def test_store_computed_features_offline_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_view_requests: list[FeatureViewRequest]
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
                feature_view_requests
            )

    def test_store_computed_features_offline_no_new_features_stored(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_view_requests: list[FeatureViewRequest]
    ) -> None:
        """Test handling when no new features are stored (duplicates)."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = 0

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_view_requests
        )

        # Then
        mock_offline_repo.store_features_incrementally.assert_called_once()
        # Should not create feature views when no new features are stored

    @patch('drl_trading_preprocess.adapter.feature_store.feature_store_save_repository.logger')
    def test_store_computed_features_offline_with_feature_views_creation(
        self,
        mock_logger: Mock,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_view_requests: list[FeatureViewRequest]
    ) -> None:
        """Test that feature views are created when new features are stored."""
        # Given
        stored_count = len(sample_features_df)
        mock_offline_repo.store_features_incrementally.return_value = stored_count

        from types import SimpleNamespace
        mock_obs_fv = SimpleNamespace(name="observation_space_feature_view")
        mock_reward_fv = SimpleNamespace(name="reward_engineering_feature_view")
        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.create_feature_view_from_request.side_effect = [mock_obs_fv, mock_reward_fv]
        mock_feast_provider.create_feature_service.return_value = mock_feature_service

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_view_requests
        )

        # Then
        # Verify feature views creation
        assert mock_feast_provider.create_feature_view_from_request.call_count == 2
        mock_feast_provider.create_feature_service.assert_called_once_with(
            feature_views=[mock_obs_fv, mock_reward_fv],
            symbol=eurusd_h1_symbol,
            feature_version_info=feature_view_requests[0].feature_version_info
        )

        # Verify feature store apply (implementation includes entity)
        call_args = feature_store_save_repository.feature_store.apply.call_args[0][0]
        # There should be 4 components: entity, observation feature view, reward feature view, and feature service
        EXPECTED_APPLY_COMPONENTS_COUNT = 4  # entity + obs_fv + reward_fv + feature_service
        assert len(call_args) == EXPECTED_APPLY_COMPONENTS_COUNT

        # Verify that the apply call includes the expected components
        # The entity is first, followed by the feature views and service
        assert mock_obs_fv in call_args
        assert mock_reward_fv in call_args
        assert mock_feature_service in call_args

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
        # Implementation converts timestamps to UTC
        expected_start = sample_features_df["event_timestamp"].min().tz_localize("UTC")
        expected_end = sample_features_df["event_timestamp"].max().tz_localize("UTC")

        feature_store_save_repository.feature_store.materialize.assert_called_once_with(
            start_date=expected_start,
            end_date=expected_end
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

    @patch('drl_trading_preprocess.adapter.feature_store.feature_store_save_repository.logger')
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
            "symbol": [eurusd_h1_symbol],
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

        # Implementation filters and sorts columns, so we need to check the actual call
        call_args = feature_store_save_repository.feature_store.write_to_online_store.call_args
        actual_feature_view_name = call_args[1]["feature_view_name"]
        actual_df = call_args[1]["df"]

        assert actual_feature_view_name == expected_feature_view_name

        # Verify DataFrame content (implementation sorts columns alphabetically)
        expected_columns = ["event_timestamp", "feature_1", "symbol"]
        assert list(actual_df.columns) == expected_columns

        # Verify data content is preserved
        assert len(actual_df) == 1
        assert actual_df["symbol"].iloc[0] == eurusd_h1_symbol
        assert actual_df["feature_1"].iloc[0] == 1.5

    def test_push_features_to_online_store_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "symbol": [eurusd_h1_symbol],
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
            "symbol": [eurusd_h1_symbol],
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

    @patch('drl_trading_preprocess.adapter.feature_store.feature_store_save_repository.logger')
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
            "symbol": [eurusd_h1_symbol],
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
        # Implementation makes multiple debug calls (one for each step: filtering columns, sorting, mapping, writing, and summary)
        EXPECTED_DEBUG_CALLS = 5  # 5 debug calls: filter, sort, map, write, summary
        assert mock_logger.debug.call_count == EXPECTED_DEBUG_CALLS

        # Verify the final debug message contains the expected information
        final_call = mock_logger.debug.call_args_list[-1]
        debug_message = final_call[0][0]
        assert "Pushed 1 feature records" in debug_message
        assert "EURUSD" in debug_message
        assert "Pushed 1 feature records" in debug_message
        assert "EURUSD" in debug_message


class TestFeatureStoreSaveRepositoryUtilityMethods:
        """Placeholder for utility method tests."""
        pass


class TestFeatureStoreSaveRepositoryPrivateMethods:
    """Test class for private methods (through public interface)."""

    def test_create_and_apply_feature_views_called_correctly(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_view_requests: list[FeatureViewRequest]
    ) -> None:
        """Test that _create_and_apply_feature_views is called with correct parameters."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        from types import SimpleNamespace
        mock_obs_fv = SimpleNamespace(name="observation_space_feature_view")
        mock_reward_fv = SimpleNamespace(name="reward_engineering_feature_view")
        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.create_feature_view_from_request.side_effect = [mock_obs_fv, mock_reward_fv]
        mock_feast_provider.create_feature_service.return_value = mock_feature_service

        # When
        feature_store_save_repository.store_computed_features_offline(
            sample_features_df,
            eurusd_h1_symbol,
            feature_view_requests
        )

        # Then
        # Verify feature views are created from requests
        calls = mock_feast_provider.create_feature_view_from_request.call_args_list
        assert len(calls) == 2
        # First and second request should be passed as the first positional arg
        assert isinstance(calls[0].args[0], FeatureViewRequest)
        assert isinstance(calls[1].args[0], FeatureViewRequest)


class TestFeatureStoreSaveRepositoryErrorHandling:
    """Test class for error handling scenarios."""

    def test_store_computed_features_offline_feast_provider_exception(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_view_requests: list[FeatureViewRequest],
    ) -> None:
        """Test error handling when feast provider raises exception."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)
        mock_feast_provider.create_feature_view_from_request.side_effect = Exception("Feast provider error")

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            feature_store_save_repository.store_computed_features_offline(
                sample_features_df,
                eurusd_h1_symbol,
                feature_view_requests,
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
            "symbol": [eurusd_h1_symbol],
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
            "symbol": [eurusd_h1_symbol],
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

        # Implementation filters and sorts columns, so we need to check the actual call
        call_args = feature_store_save_repository.feature_store.write_to_online_store.call_args
        actual_feature_view_name = call_args[1]["feature_view_name"]
        actual_df = call_args[1]["df"]

        assert actual_feature_view_name == expected_view_name

        # Verify DataFrame content (implementation sorts columns alphabetically)
        expected_columns = ["event_timestamp", "feature_1", "symbol"]
        assert list(actual_df.columns) == expected_columns

        # Verify data content is preserved
        assert len(actual_df) == 1
        assert actual_df["symbol"].iloc[0] == eurusd_h1_symbol
        assert actual_df["feature_1"].iloc[0] == 1.5
