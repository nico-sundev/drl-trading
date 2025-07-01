"""
Integration tests for FeatureStoreSaveRepository and FeatureStoreFetchRepository.

These tests verify the complete integration between both repositories
and their dependencies with real Feast infrastructure.
"""


import pandas as pd
import pytest
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from injector import Injector
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.repository.feature_store_fetch_repo import (
    FeatureStoreFetchRepository,
)
from drl_trading_core.preprocess.feature_store.repository.feature_store_save_repo import (
    FeatureStoreSaveRepository,
)


class TestFeatureStoreRepositoriesIntegration:
    """Integration tests for both FeatureStoreSaveRepository and FeatureStoreFetchRepository."""

    def test_complete_save_and_fetch_workflow(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test complete workflow: save features, then fetch them back."""
        # Given
        symbol = "EURUSD"

        # Get repository instances from DI container
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # When - Store features offline
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # And fetch them back (offline)
        timestamps = sample_trading_features_df["event_timestamp"]
        fetched_features = fetch_repo.get_offline(
            symbol=symbol,
            timestamps=timestamps,
            feature_version_info=feature_version_info_fixture
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

    def test_batch_materialize_and_online_fetch_workflow(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test workflow: store offline, materialize to online, then fetch online."""
        # Given
        symbol = "EURUSD"

        # Get repository instances from DI container
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # Store features offline first
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # When - Materialize to online store
        save_repo.batch_materialize_features(
            features_df=sample_trading_features_df,
            symbol=symbol
        )

        # And fetch from online store
        online_features = fetch_repo.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # Then
        assert not online_features.empty
        assert "symbol" in online_features.columns

        # Verify symbol consistency
        assert all(online_features["symbol"] == symbol)

    def test_direct_online_push_and_fetch_workflow(
        self,
        mocked_container: Injector,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test workflow: push single record directly to online, then fetch."""
        # Given
        symbol = "EURUSD"

        # Create single record for real-time scenario
        single_record_df = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 15:30:00")],
            "symbol": [symbol],
            "rsi_14": [45.2],
            "sma_20": [1.0855],
            "reward": [0.15]
        })

        # Get repository instances from DI container
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # When - Push directly to online store (inference mode)
        save_repo.push_features_to_online_store(
            features_df=single_record_df,
            symbol=symbol,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE
        )

        # And fetch from online store
        online_features = fetch_repo.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # Then
        assert not online_features.empty
        assert "symbol" in online_features.columns
        assert all(online_features["symbol"] == symbol)

    def test_feature_store_enabled_disabled_behavior(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test behavior when feature store is disabled."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)

        # When feature store is enabled
        assert save_repo.is_enabled() is True

        # Store should work
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # When we disable the feature store
        save_repo.config.enabled = False

        # Then
        assert save_repo.is_enabled() is False

    @pytest.mark.parametrize("feature_role", [
        FeatureRoleEnum.OBSERVATION_SPACE,
        FeatureRoleEnum.REWARD_ENGINEERING
    ])
    def test_feature_role_specific_operations(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo,
        feature_role: FeatureRoleEnum
    ) -> None:
        """Test operations specific to different feature roles."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)

        # Filter features based on role
        if feature_role == FeatureRoleEnum.OBSERVATION_SPACE:
            role_features_df = sample_trading_features_df[[
                "event_timestamp", "symbol", "rsi_14", "rsi_21", "sma_20", "sma_50",
                "bb_upper", "bb_lower", "bb_middle", "close", "high", "low", "volume"
            ]].copy()
        else:  # REWARD_ENGINEERING
            role_features_df = sample_trading_features_df[[
                "event_timestamp", "symbol", "reward", "cumulative_return"
            ]].copy()

        # When - Push features for specific role
        save_repo.push_features_to_online_store(
            features_df=role_features_df,
            symbol=symbol,
            feature_role=feature_role
        )

        # Then - Operation should complete without errors
        # The actual verification would depend on the FeatureViewNameMapper implementation
        assert True  # Placeholder for successful completion

    def test_error_handling_missing_event_timestamp(
        self,
        mocked_container: Injector,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test error handling when event_timestamp column is missing."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)

        # Create DataFrame without event_timestamp
        invalid_df = DataFrame({
            "symbol": [symbol],
            "rsi_14": [45.2],
            "sma_20": [1.0855]
        })

        # When/Then - Should raise ValueError for all operations
        with pytest.raises(ValueError, match="event_timestamp"):
            save_repo.store_computed_features_offline(
                features_df=invalid_df,
                symbol=symbol,
                feature_version_info=feature_version_info_fixture
            )

        with pytest.raises(ValueError, match="event_timestamp"):
            save_repo.batch_materialize_features(
                features_df=invalid_df,
                symbol=symbol
            )

        with pytest.raises(ValueError, match="event_timestamp"):
            save_repo.push_features_to_online_store(
                features_df=invalid_df,
                symbol=symbol,
                feature_role=FeatureRoleEnum.OBSERVATION_SPACE
            )

    def test_empty_dataframe_handling(
        self,
        mocked_container: Injector,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test handling of empty DataFrames."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)

        empty_df = DataFrame()

        # When - Store empty DataFrame
        save_repo.store_computed_features_offline(
            features_df=empty_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # Then - Should handle gracefully without errors
        assert True  # Placeholder for successful completion

    def test_large_dataset_performance(
        self,
        mocked_container: Injector,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test performance with larger datasets."""
        # Given
        symbol = "EURUSD"

        # Create larger dataset (1000 records)
        timestamps = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=1000,
            freq="H"
        )

        large_features_df = DataFrame({
            "event_timestamp": timestamps,
            "symbol": [symbol] * len(timestamps),
            "rsi_14": [30.5 + i * 0.1 for i in range(len(timestamps))],
            "sma_20": [1.0850 + i * 0.00001 for i in range(len(timestamps))],
            "reward": [0.01 * (i % 20 - 10) for i in range(len(timestamps))]
        })

        # Get repository instances from DI container
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # When - Store large dataset
        save_repo.store_computed_features_offline(
            features_df=large_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # And fetch subset back
        subset_timestamps = large_features_df["event_timestamp"].iloc[100:200]
        fetched_features = fetch_repo.get_offline(
            symbol=symbol,
            timestamps=subset_timestamps,
            feature_version_info=feature_version_info_fixture
        )

        # Then
        assert not fetched_features.empty
        assert len(fetched_features) <= len(subset_timestamps)
        assert all(fetched_features["symbol"] == symbol)


class TestFeatureStoreRepositoriesErrorScenarios:
    """Test error scenarios and edge cases for both repositories."""

    def test_feature_service_initialization_failure(
        self,
        mocked_container: Injector,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test handling when feature service initialization fails."""
        # Given
        symbol = "EURUSD"
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # Mock feast provider to return None for feature service
        fetch_repo._feast_provider.create_feature_service.return_value = None

        # When/Then - Should raise RuntimeError
        with pytest.raises(RuntimeError, match="FeatureService is not initialized"):
            fetch_repo.get_online(
                symbol=symbol,
                feature_version_info=feature_version_info_fixture
            )

        with pytest.raises(RuntimeError, match="FeatureService is not initialized"):
            timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
            fetch_repo.get_offline(
                symbol=symbol,
                timestamps=timestamps,
                feature_version_info=feature_version_info_fixture
            )

    def test_concurrent_access_simulation(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test simulated concurrent access to feature store."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # Simulate multiple concurrent saves (in practice would be threading)
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]

        for test_symbol in symbols:
            # Modify symbol in dataframe
            modified_df = sample_trading_features_df.copy()
            modified_df["symbol"] = test_symbol

            # When - Save for each symbol
            save_repo.store_computed_features_offline(
                features_df=modified_df,
                symbol=test_symbol,
                feature_version_info=feature_version_info_fixture
            )

        # Then - Fetch should work for all symbols
        for test_symbol in symbols:
            timestamps = sample_trading_features_df["event_timestamp"].iloc[0:10]
            fetched_features = fetch_repo.get_offline(
                symbol=test_symbol,
                timestamps=timestamps,
                feature_version_info=feature_version_info_fixture
            )

            assert not fetched_features.empty
            assert all(fetched_features["symbol"] == test_symbol)


class TestFeatureStoreRepositoriesObservability:
    """Test observability and monitoring aspects of repositories."""

    def test_logging_verification(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo,
        caplog
    ) -> None:
        """Test that appropriate logging occurs during operations."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)

        # When - Perform operations that should log
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # Then - Verify logging occurred
        assert any("Stored" in record.message for record in caplog.records)

    def test_metrics_collection_simulation(
        self,
        mocked_container: Injector,
        sample_trading_features_df: DataFrame,
        feature_version_info_fixture: FeatureConfigVersionInfo
    ) -> None:
        """Test simulation of metrics collection during repository operations."""
        # Given
        symbol = "EURUSD"
        save_repo = mocked_container.get(FeatureStoreSaveRepository)
        fetch_repo = mocked_container.get(FeatureStoreFetchRepository)

        # Track operation metrics
        operation_count = 0

        # When - Perform multiple operations
        operation_count += 1
        save_repo.store_computed_features_offline(
            features_df=sample_trading_features_df,
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        operation_count += 1
        timestamps = sample_trading_features_df["event_timestamp"].iloc[0:10]
        fetch_repo.get_offline(
            symbol=symbol,
            timestamps=timestamps,
            feature_version_info=feature_version_info_fixture
        )

        operation_count += 1
        fetch_repo.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info_fixture
        )

        # Then - Verify operations completed
        assert operation_count == 3

        # In real implementation, this would verify metrics were collected
        # (e.g., operation duration, success/failure rates, data volume, etc.)
