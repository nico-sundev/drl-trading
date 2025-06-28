"""
Unit tests for FeatureStoreFetchRepository.

Tests the feature store fetch logic with mocked dependencies
to isolate the business logic from external infrastructure.
"""

from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import FeatureService, FeatureStore
from pandas import DataFrame

from drl_trading_core.preprocess.feature_store.repository.feature_store_fetch_repo import (
    FeatureStoreFetchRepository,
    IFeatureStoreFetchRepository,
)


class TestFeatureStoreFetchRepositoryInit:
    """Test class for FeatureStoreFetchRepository initialization."""

    def test_init_with_valid_dependencies(
        self,
        mock_feast_provider: Mock
    ) -> None:
        """Test successful initialization with valid dependencies."""
        # Given
        mock_feature_store = Mock(spec=FeatureStore)

        # When
        repo = FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

        # Then
        assert repo._fs == mock_feature_store
        assert repo._feast_provider == mock_feast_provider
        assert repo._feature_service is None

    def test_implements_interface(
        self,
        mock_feast_provider: Mock
    ) -> None:
        """Test that FeatureStoreFetchRepository implements the correct interface."""
        # Given
        mock_feature_store = Mock(spec=FeatureStore)

        # When
        repo = FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

        # Then
        assert isinstance(repo, IFeatureStoreFetchRepository)


class TestFeatureStoreFetchRepositoryGetOnline:
    """Test class for get_online method."""

    @pytest.fixture
    def repository(
        self,
        mock_feature_store: Mock,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        return FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

    def test_get_online_first_call_creates_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online creates feature service on first call."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "EURUSD"
        expected_entity_rows = [{"symbol": symbol}]

        # When
        result = repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_called_once_with(
            symbol=symbol,
            feature_version_info=feature_version_info
        )
        mock_feature_store.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)
        assert repository._feature_service == mock_feature_service

    def test_get_online_reuses_existing_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online reuses existing feature service on subsequent calls."""
        # Given
        repository._feature_service = mock_feature_service
        symbol = "EURUSD"
        expected_entity_rows = [{"symbol": symbol}]

        # When
        result = repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_not_called()
        mock_feature_store.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)

    def test_get_online_raises_error_when_feature_service_is_none(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online raises error when feature service creation returns None."""
        # Given
        mock_feast_provider.create_feature_service.return_value = None
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(RuntimeError, match="FeatureService is not initialized. Cannot fetch online features."):
            repository.get_online(
                symbol=symbol,
                feature_version_info=feature_version_info
            )

    def test_get_online_with_different_symbol(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test get_online with different symbol."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "GBPUSD"
        expected_entity_rows = [{"symbol": symbol}]

        # When
        result = repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feature_store.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)

    def test_get_online_returns_dataframe_from_feast_response(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online returns the DataFrame from Feast response."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "EURUSD"
        expected_df = DataFrame({
            "symbol": [symbol],
            "rsi_14": [45.2],
            "sma_20": [1.0855],
            "event_timestamp": [pd.Timestamp("2024-01-01 10:00:00")]
        })
        mock_response = Mock()
        mock_response.to_df.return_value = expected_df
        mock_feature_store.get_online_features.return_value = mock_response

        # When
        result = repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        pd.testing.assert_frame_equal(result, expected_df)


class TestFeatureStoreFetchRepositoryGetOffline:
    """Test class for get_offline method."""

    @pytest.fixture
    def repository(
        self,
        mock_feature_store: Mock,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        return FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

    @pytest.fixture
    def sample_timestamps(self) -> pd.Series:
        """Create sample timestamps for testing."""
        return pd.Series([
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ])

    def test_get_offline_first_call_creates_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline creates feature service on first call."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "EURUSD"

        # When
        result = repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_called_once_with(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Verify entity_df structure
        call_args = mock_feature_store.get_historical_features.call_args
        assert call_args[1]["features"] == mock_feature_service
        entity_df = call_args[1]["entity_df"]

        pd.testing.assert_series_equal(entity_df["event_timestamp"], sample_timestamps, check_names=False)
        assert all(entity_df["symbol"] == symbol)

        assert isinstance(result, DataFrame)
        assert repository._feature_service == mock_feature_service

    def test_get_offline_reuses_existing_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline reuses existing feature service on subsequent calls."""
        # Given
        repository._feature_service = mock_feature_service
        symbol = "EURUSD"

        # When
        result = repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_not_called()
        mock_feature_store.get_historical_features.assert_called_once()
        assert isinstance(result, DataFrame)

    def test_get_offline_raises_error_when_feature_service_is_none(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline raises error when feature service creation returns None."""
        # Given
        mock_feast_provider.create_feature_service.return_value = None
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(RuntimeError, match="FeatureService is not initialized. Cannot fetch online features."):
            repository.get_offline(
                symbol=symbol,
                timestamps=sample_timestamps,
                feature_version_info=feature_version_info
            )

    def test_get_offline_with_different_symbol(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo,
        sample_timestamps: pd.Series
    ) -> None:
        """Test get_offline with different symbol."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "GBPUSD"

        # When
        result = repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        call_args = mock_feature_store.get_historical_features.call_args
        entity_df = call_args[1]["entity_df"]

        assert all(entity_df["symbol"] == symbol)
        assert isinstance(result, DataFrame)

    def test_get_offline_with_empty_timestamps(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test get_offline with empty timestamps series."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        empty_timestamps = pd.Series([], dtype='datetime64[ns]')
        symbol = "EURUSD"

        # When
        result = repository.get_offline(
            symbol=symbol,
            timestamps=empty_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        call_args = mock_feature_store.get_historical_features.call_args
        entity_df = call_args[1]["entity_df"]

        assert len(entity_df) == 0
        assert "event_timestamp" in entity_df.columns
        assert "symbol" in entity_df.columns
        assert isinstance(result, DataFrame)

    def test_get_offline_returns_dataframe_from_feast_response(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feature_store: Mock,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_version_info: FeatureConfigVersionInfo,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline returns the DataFrame from Feast response."""
        # Given
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "EURUSD"
        expected_df = DataFrame({
            "event_timestamp": sample_timestamps,
            "symbol": [symbol] * 3,
            "rsi_14": [30.5, 45.2, 67.8],
            "sma_20": [1.0850, 1.0855, 1.0860]
        })
        mock_response = Mock()
        mock_response.to_df.return_value = expected_df
        mock_feature_store.get_historical_features.return_value = mock_response

        # When
        result = repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        pd.testing.assert_frame_equal(result, expected_df)


class TestFeatureStoreFetchRepositoryErrorHandling:
    """Test class for error handling scenarios."""

    @pytest.fixture
    def repository(
        self,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        mock_feature_store = Mock(spec=FeatureStore)
        return FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

    def test_get_online_handles_feast_provider_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online properly propagates FeastProvider exceptions."""
        # Given
        mock_feast_provider.create_feature_service.side_effect = Exception("Feast provider error")
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            repository.get_online(
                symbol=symbol,
                feature_version_info=feature_version_info
            )

    def test_get_offline_handles_feast_provider_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_offline properly propagates FeastProvider exceptions."""
        # Given
        mock_feast_provider.create_feature_service.side_effect = Exception("Feast provider error")
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            repository.get_offline(
                symbol=symbol,
                timestamps=sample_timestamps,
                feature_version_info=feature_version_info
            )

    def test_get_online_handles_feature_store_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_online properly propagates FeatureStore exceptions."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        repository._fs.get_online_features.side_effect = Exception("Feature store error")
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(Exception, match="Feature store error"):
            repository.get_online(
                symbol=symbol,
                feature_version_info=feature_version_info
            )

    def test_get_offline_handles_feature_store_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that get_offline properly propagates FeatureStore exceptions."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        repository._fs.get_historical_features.side_effect = Exception("Feature store error")
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        symbol = "EURUSD"

        # When & Then
        with pytest.raises(Exception, match="Feature store error"):
            repository.get_offline(
                symbol=symbol,
                timestamps=sample_timestamps,
                feature_version_info=feature_version_info
            )


class TestFeatureStoreFetchRepositoryFeatureServiceCaching:
    """Test class for feature service caching behavior."""

    @pytest.fixture
    def repository(
        self,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        mock_feature_store = Mock(spec=FeatureStore)
        mock_response = Mock()
        mock_response.to_df.return_value = DataFrame({
            "symbol": ["EURUSD"],
            "rsi_14": [45.2]
        })
        mock_feature_store.get_online_features.return_value = mock_response
        mock_feature_store.get_historical_features.return_value = mock_response
        return FeatureStoreFetchRepository(
            feature_store=mock_feature_store,
            feast_provider=mock_feast_provider
        )

    def test_feature_service_created_once_for_multiple_online_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that feature service is created only once for multiple online calls."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        symbol = "EURUSD"

        # When
        repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )
        repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )
        repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_called_once()

    def test_feature_service_created_once_for_multiple_offline_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that feature service is created only once for multiple offline calls."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        symbol = "EURUSD"

        # When
        repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )
        repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )
        repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_called_once()

    def test_feature_service_shared_between_online_and_offline_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_version_info: FeatureConfigVersionInfo
    ) -> None:
        """Test that feature service is shared between online and offline calls."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.create_feature_service.return_value = mock_feature_service
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        symbol = "EURUSD"

        # When
        repository.get_online(
            symbol=symbol,
            feature_version_info=feature_version_info
        )
        repository.get_offline(
            symbol=symbol,
            timestamps=sample_timestamps,
            feature_version_info=feature_version_info
        )

        # Then
        mock_feast_provider.create_feature_service.assert_called_once()
        assert repository._feature_service == mock_feature_service
