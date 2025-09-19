"""
Parametrized integration tests for FeatureStoreSaveRepository and FeatureStoreFetchRepository.

These tests verify that both repositories work correctly with different offline
repository strategies (local filesystem and S3).
"""

import logging
from drl_trading_core.common.model.feature_view_request_container import FeatureViewRequestContainer
from drl_trading_core.common.model.feature_service_request_container import (
    FeatureServiceRequestContainer,
)
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from injector import Injector
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.feature_store_fetch_repository import (
    IFeatureStoreFetchPort,
)
from drl_trading_preprocess.core.port.feature_store_save_port import IFeatureStoreSavePort

logger = logging.getLogger(__name__)


class TestParametrizedFeatureStoreRepositoriesIntegration:
    """Parametrized integration tests for both repositories with different offline strategies."""

    def test_complete_save_and_fetch_workflow_parametrized(
        self,
        parametrized_integration_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_view_requests_fixture: list[FeatureViewRequestContainer],
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test complete workflow with both local and S3 offline repositories."""
        # Given
        symbol = "EURUSD"

        # Get repository instances from DI container
        save_repo = parametrized_integration_container.get(IFeatureStoreSavePort)
        fetch_repo = parametrized_integration_container.get(IFeatureStoreFetchPort)

        # When - Store features offline
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # And fetch them back (offline)
        timestamps = sample_trading_features_df["event_timestamp"]
        feature_service_request = FeatureServiceRequestContainer(
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            timeframe=Timeframe.HOUR_1
        )
        fetched_features = fetch_repo.get_offline(
            feature_service_request=feature_service_request,
            timestamps=timestamps
        )

        # Then
        assert not fetched_features.empty
        assert "symbol" in fetched_features.columns
        assert "event_timestamp" in fetched_features.columns
        assert len(fetched_features) > 0

        # Verify symbol consistency
        assert all(fetched_features["symbol"] == symbol)

        # Verify timestamp range
        min_timestamp = fetched_features["event_timestamp"].min()
        max_timestamp = fetched_features["event_timestamp"].max()
        assert min_timestamp >= timestamps.min()
        assert max_timestamp <= timestamps.max()

    def test_batch_materialize_and_online_fetch_workflow_parametrized(
        self,
        parametrized_integration_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_view_requests_fixture: list[FeatureViewRequestContainer],
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test workflow with both offline strategies: store offline, materialize to online, then fetch online."""
        # Given
        symbol = "EURUSD"

        # Get repository instances from DI container
        save_repo = parametrized_integration_container.get(IFeatureStoreSavePort)
        fetch_repo = parametrized_integration_container.get(IFeatureStoreFetchPort)

        # Store features offline first
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # When - Materialize to online store
        save_repo.batch_materialize_features(
            features_df=sample_trading_features_df,
            symbol=symbol
        )

        # And fetch from online store
        feature_service_request = FeatureServiceRequestContainer(
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            timeframe=Timeframe.HOUR_1
        )
        online_features = fetch_repo.get_online(
            feature_service_request=feature_service_request
        )

        # Then
        assert not online_features.empty
        assert "symbol" in online_features.columns
        assert len(online_features) > 0

        # Verify symbol consistency
        assert all(online_features["symbol"] == symbol)

    def test_handle_duplicate_features_parametrized(
        self,
        parametrized_integration_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_view_requests_fixture: list[FeatureViewRequestContainer],
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test that duplicate features are handled correctly with both offline strategies."""
        # Given
        symbol = "EURUSD"
        save_repo = parametrized_integration_container.get(IFeatureStoreSavePort)
        fetch_repo = parametrized_integration_container.get(IFeatureStoreFetchPort)

        # Store initial features
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # When - Store the same features again (should handle duplicates gracefully)
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # Then - Fetch should still return the correct number of unique features
        timestamps = sample_trading_features_df["event_timestamp"]
        feature_service_request = FeatureServiceRequestContainer(
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            timeframe=Timeframe.HOUR_1
        )
        fetched_features = fetch_repo.get_offline(
            feature_service_request=feature_service_request,
            timestamps=timestamps
        )

        assert not fetched_features.empty
        assert len(fetched_features) <= len(sample_trading_features_df)

        # Verify no duplicate timestamps
        assert len(fetched_features["event_timestamp"].unique()) == len(fetched_features)

    def test_incremental_feature_storage_parametrized(
        self,
        parametrized_integration_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_view_requests_fixture: list[FeatureViewRequestContainer],
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test incremental feature storage with both offline strategies."""
        # Given
        symbol = "EURUSD"
        save_repo = parametrized_integration_container.get(IFeatureStoreSavePort)
        fetch_repo = parametrized_integration_container.get(IFeatureStoreFetchPort)

        # Split the data into two batches
        mid_point = len(sample_trading_features_df) // 2
        first_batch = sample_trading_features_df.iloc[:mid_point]
        second_batch = sample_trading_features_df.iloc[mid_point:]

        # When - Store first batch
        save_repo.store_computed_features_offline(
            features_df=first_batch,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # And store second batch
        save_repo.store_computed_features_offline(
            features_df=second_batch,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            feature_view_requests=feature_view_requests_fixture
        )

        # Then - Fetch all features
        all_timestamps = sample_trading_features_df["event_timestamp"]
        feature_service_request = FeatureServiceRequestContainer(
            feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture,
            timeframe=Timeframe.HOUR_1
        )
        fetched_features = fetch_repo.get_offline(
            feature_service_request=feature_service_request,
            timestamps=all_timestamps
        )

        assert not fetched_features.empty
        assert len(fetched_features) == len(sample_trading_features_df)

        # Verify all timestamps are present
        fetched_timestamps = set(fetched_features["event_timestamp"])
        expected_timestamps = set(sample_trading_features_df["event_timestamp"])
        assert fetched_timestamps == expected_timestamps

    def test_multiple_symbols_parametrized(
        self,
        parametrized_integration_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test storing and fetching features for multiple symbols with both offline strategies."""
        # Given
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        save_repo = parametrized_integration_container.get(IFeatureStoreSavePort)
        fetch_repo = parametrized_integration_container.get(IFeatureStoreFetchPort)

        # When - Store features for multiple symbols
        for symbol in symbols:
            symbol_features = sample_trading_features_df.copy()
            symbol_features["symbol"] = symbol

            # Import the factory function to create feature view requests for this specific symbol
            from conftest import create_feature_view_requests
            symbol_feature_requests = create_feature_view_requests(feature_version_info_fixture, symbol=symbol)

            save_repo.store_computed_features_offline(
                features_df=symbol_features,
                symbol=symbol,
                feature_version_info=feature_version_info_fixture,
                feature_view_requests=symbol_feature_requests
            )

        # Then - Fetch features for each symbol and verify isolation
        timestamps = sample_trading_features_df["event_timestamp"]
        for symbol in symbols:
            feature_service_request = FeatureServiceRequestContainer(
                feature_service_role=FeatureRoleEnum.OBSERVATION_SPACE,
                symbol=symbol,
                feature_version_info=feature_version_info_fixture,
                timeframe=Timeframe.HOUR_1
            )
            fetched_features = fetch_repo.get_offline(
                feature_service_request=feature_service_request,
                timestamps=timestamps
            )

            assert not fetched_features.empty, f"No features fetched for symbol {symbol}"
            assert all(fetched_features["symbol"] == symbol)
            assert len(fetched_features) == len(sample_trading_features_df)
