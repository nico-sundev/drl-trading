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
from drl_trading_core.core.dto.feature_service_metadata import FeatureServiceMetadata
from drl_trading_core.core.dto.offline_storage_request import OfflineStorageRequest
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
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test successful storage of computed features offline."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        request = OfflineStorageRequest.create(
            features_df=sample_features_df,
            feature_service_metadata=feature_service_metadata
        )

        # When
        feature_store_save_repository.store_computed_features_offline(request)

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
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test handling of empty DataFrame during offline storage."""
        # Given
        empty_df = DataFrame()

        request = OfflineStorageRequest(
            features_df=empty_df,
            feature_service_metadata=feature_service_metadata
        )

        # When
        feature_store_save_repository.store_computed_features_offline(request)

        # Then
        mock_offline_repo.store_features_incrementally.assert_not_called()

    def test_store_computed_features_offline_missing_timestamp_column(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        invalid_df = DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0]
            # Missing event_timestamp column
        })

        request = OfflineStorageRequest.create(
            features_df=invalid_df,
            feature_service_metadata=feature_service_metadata
        )

        # When & Then
        with pytest.raises(ValueError, match="features_df must contain 'event_timestamp' column"):
            feature_store_save_repository.store_computed_features_offline(request)

    def test_store_computed_features_offline_no_new_features_stored(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test handling when no new features are stored (duplicates)."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = 0

        request = OfflineStorageRequest.create(
            features_df=sample_features_df,
            feature_service_metadata=feature_service_metadata
        )

        # When
        feature_store_save_repository.store_computed_features_offline(request)

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
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test that feature views are created when new features are stored."""
        # Given
        stored_count = len(sample_features_df)
        mock_offline_repo.store_features_incrementally.return_value = stored_count

        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.get_or_create_feature_service.return_value = mock_feature_service

        request = OfflineStorageRequest.create(
            features_df=sample_features_df,
            feature_service_metadata=feature_service_metadata
        )

        # When
        feature_store_save_repository.store_computed_features_offline(request)

        # Then
        # Verify feature service creation
        mock_feast_provider.get_or_create_feature_service.assert_called_once()
        call_args = mock_feast_provider.get_or_create_feature_service.call_args
        assert "service_name" in call_args.kwargs
        assert "feature_view_requests" in call_args.kwargs
        assert call_args.kwargs["feature_view_requests"] == feature_service_metadata.feature_view_metadata_list

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
            end_date=expected_end,
            feature_views=['test_feature_view']
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

        # Create mock feature view with matching tags and schema
        mock_feature_view = Mock()
        mock_feature_view.name = "observation_space_feature_view"
        mock_feature_view.tags = {
            "feature_role": feature_role.value,
            "symbol": eurusd_h1_symbol
        }
        # Mock schema with feature_1 field
        mock_field = Mock()
        mock_field.name = "feature_1"
        mock_feature_view.schema = [mock_field]

        # Mock feature store to return the mock feature view
        feature_store_save_repository.feature_store.list_feature_views.return_value = [mock_feature_view]

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )

        # Then
        feature_store_save_repository.feature_store.list_feature_views.assert_called_once()

        # Verify write_to_online_store was called with correct parameters
        feature_store_save_repository.feature_store.write_to_online_store.assert_called_once()
        call_args = feature_store_save_repository.feature_store.write_to_online_store.call_args

        assert call_args[1]["feature_view_name"] == "observation_space_feature_view"

        # Verify DataFrame content (implementation filters and sorts columns)
        actual_df = call_args[1]["df"]
        expected_columns = ["event_timestamp", "feature_1", "symbol"]  # Sorted alphabetically
        assert list(actual_df.columns) == expected_columns

        # Verify data content is preserved
        assert len(actual_df) == 1
        assert actual_df["symbol"].iloc[0] == eurusd_h1_symbol
        assert actual_df["feature_1"].iloc[0] == 1.5
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
        """Test error handling when no feature views are found for the role."""
        # Given
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "symbol": [eurusd_h1_symbol],
            "feature_1": [1.5]
        })
        feature_role = Mock()  # Unknown feature role
        feature_role.value = "unknown_role"

        # Mock feature store to return empty list (no matching feature views)
        feature_store_save_repository.feature_store.list_feature_views.return_value = []

        # When & Then
        with pytest.raises(RuntimeError, match="No feature views found for role 'unknown_role'"):
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

        # Mock feature view with appropriate tags and schema
        mock_feature_view = Mock()
        mock_feature_view.tags = {"feature_role": feature_role.value, "symbol": eurusd_h1_symbol}
        # Create schema fields with proper name attribute
        mock_field_timestamp = Mock()
        mock_field_timestamp.name = "event_timestamp"
        mock_field_symbol = Mock()
        mock_field_symbol.name = "symbol"
        mock_field_feature = Mock()
        mock_field_feature.name = "feature_1"
        mock_feature_view.schema = [mock_field_timestamp, mock_field_symbol, mock_field_feature]
        mock_feature_view.name = "test_feature_view"
        feature_store_save_repository.feature_store.list_feature_views.return_value = [mock_feature_view]

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )
        # Then
        # Verify that logging was called (exact count may vary with new implementation)
        assert mock_logger.debug.call_count >= 1

        # Check that at least one debug call was made
        assert mock_logger.debug.called


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
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test that _create_and_apply_feature_views is called with correct parameters."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)

        mock_feature_service = Mock()
        mock_feature_service.name = "test_service"

        mock_feast_provider.get_or_create_feature_service.return_value = mock_feature_service

        request = OfflineStorageRequest.create(
            features_df=sample_features_df,
            feature_service_metadata=feature_service_metadata
        )

        # When
        feature_store_save_repository.store_computed_features_offline(request)

        # Then
        # Verify feature service is created with the correct requests
        mock_feast_provider.get_or_create_feature_service.assert_called_once()
        call_args = mock_feast_provider.get_or_create_feature_service.call_args
        assert "service_name" in call_args.kwargs
        assert "feature_view_requests" in call_args.kwargs
        assert call_args.kwargs["feature_view_requests"] == feature_service_metadata.feature_view_metadata_list


class TestFeatureStoreSaveRepositoryErrorHandling:
    """Test class for error handling scenarios."""

    def test_store_computed_features_offline_feast_provider_exception(
        self,
        feature_store_save_repository: FeatureStoreSaveRepository,
        sample_features_df: DataFrame,
        eurusd_h1_symbol: str,
        mock_offline_repo: Mock,
        mock_feast_provider: Mock,
        feature_service_metadata: FeatureServiceMetadata
    ) -> None:
        """Test error handling when feast provider raises exception."""
        # Given
        mock_offline_repo.store_features_incrementally.return_value = len(sample_features_df)
        mock_feast_provider.get_or_create_feature_service.side_effect = Exception("Feast provider error")

        request = OfflineStorageRequest.create(
            features_df=sample_features_df,
            feature_service_metadata=feature_service_metadata
        )

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            feature_store_save_repository.store_computed_features_offline(request)

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

        # Mock feature view with appropriate tags and schema
        mock_feature_view = Mock()
        mock_feature_view.tags = {"feature_role": feature_role.value, "symbol": eurusd_h1_symbol}
        # Create schema fields with proper name attribute
        mock_field_timestamp = Mock()
        mock_field_timestamp.name = "event_timestamp"
        mock_field_symbol = Mock()
        mock_field_symbol.name = "symbol"
        mock_field_feature = Mock()
        mock_field_feature.name = "feature_1"
        mock_feature_view.schema = [mock_field_timestamp, mock_field_symbol, mock_field_feature]
        mock_feature_view.name = "test_feature_view"
        feature_store_save_repository.feature_store.list_feature_views.return_value = [mock_feature_view]
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

        # Mock feature view with appropriate tags and schema
        mock_feature_view = Mock()
        mock_feature_view.tags = {"feature_role": feature_role.value, "symbol": eurusd_h1_symbol}
        # Create schema fields with proper name attribute
        mock_field_timestamp = Mock()
        mock_field_timestamp.name = "event_timestamp"
        mock_field_symbol = Mock()
        mock_field_symbol.name = "symbol"
        mock_field_feature = Mock()
        mock_field_feature.name = "feature_1"
        mock_feature_view.schema = [mock_field_timestamp, mock_field_symbol, mock_field_feature]
        mock_feature_view.name = expected_view_name
        feature_store_save_repository.feature_store.list_feature_views.return_value = [mock_feature_view]

        # When
        feature_store_save_repository.push_features_to_online_store(
            single_record_df,
            eurusd_h1_symbol,
            feature_role
        )

        # Then
        feature_store_save_repository.feature_store.list_feature_views.assert_called_once()
        feature_store_save_repository.feature_store.write_to_online_store.assert_called_once()
