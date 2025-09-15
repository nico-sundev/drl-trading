"""
Unit tests for FeatureStoreFetchRepository.

Tests the feature store fetch logic with mocked dependencies
to isolate the business logic from external infrastructure.
"""

from unittest.mock import Mock

import pandas as pd
import pytest
from drl_trading_core.common.model.feature_service_request_container import (
    FeatureServiceRequestContainer,
)
from feast import FeatureService, FeatureStore
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.feature_store_fetch_repository import (
    FeatureStoreFetchRepository
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
        mock_feast_provider.get_feature_store.return_value = mock_feature_store

        # When
        repo = FeatureStoreFetchRepository(feast_provider=mock_feast_provider)

        # Then
        assert repo._fs == mock_feature_store
        assert repo._feast_provider == mock_feast_provider
        mock_feast_provider.get_feature_store.assert_called_once()



class TestFeatureStoreFetchRepositoryGetOnline:
    """Test class for get_online method."""

    @pytest.fixture
    def repository(
        self,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        mock_feature_store = Mock(spec=FeatureStore)
        mock_feast_provider.get_feature_store.return_value = mock_feature_store

        # Set up default mock responses to return actual DataFrames
        mock_online_response = Mock()
        mock_online_response.to_df.return_value = DataFrame({
            "symbol": ["EURUSD"],
            "rsi_14": [45.2]
        })
        mock_feature_store.get_online_features.return_value = mock_online_response

        return FeatureStoreFetchRepository(feast_provider=mock_feast_provider)

    def test_get_online_first_call_creates_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online creates feature service on first call."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_entity_rows = [{"symbol": feature_service_request.symbol}]

        # When
        result = repository.get_online(
            feature_service_request=feature_service_request
        )

        # Then
        mock_feast_provider.get_feature_service.assert_called_once()
        repository._fs.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)

    def test_get_online_reuses_existing_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online reuses existing feature service on subsequent calls."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_entity_rows = [{"symbol": feature_service_request.symbol}]

        # When
        result = repository.get_online(
            feature_service_request=feature_service_request
        )

        # Then
        mock_feast_provider.get_feature_service.assert_called_once()
        repository._fs.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)

    def test_get_online_raises_error_when_feature_service_is_none(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online raises error when feature service is not found."""
        # Given
        mock_feast_provider.get_feature_service.side_effect = ValueError("Feature service not found")

        # When & Then
        with pytest.raises(ValueError, match="Feature service not found"):
            repository.get_online(
                feature_service_request=feature_service_request
            )

    def test_get_online_with_different_symbol(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        gbpusd_feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test get_online with different symbol."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_entity_rows = [{"symbol": gbpusd_feature_service_request.symbol}]

        # When
        result = repository.get_online(
            feature_service_request=gbpusd_feature_service_request
        )

        # Then
        repository._fs.get_online_features.assert_called_once_with(
            features=mock_feature_service,
            entity_rows=expected_entity_rows
        )
        assert isinstance(result, DataFrame)

    def test_get_online_returns_dataframe_from_feast_response(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online returns the DataFrame from Feast response."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_df = DataFrame({
            "symbol": [feature_service_request.symbol],
            "rsi_14": [45.2],
            "sma_20": [1.0855],
            "event_timestamp": [pd.Timestamp("2024-01-01 10:00:00")]
        })
        mock_response = Mock()
        mock_response.to_df.return_value = expected_df
        repository._fs.get_online_features.return_value = mock_response

        # When
        result = repository.get_online(
            feature_service_request=feature_service_request
        )

        # Then
        pd.testing.assert_frame_equal(result, expected_df)


class TestFeatureStoreFetchRepositoryGetOffline:
    """Test class for get_offline method."""

    @pytest.fixture
    def repository(
        self,
        mock_feast_provider: Mock
    ) -> FeatureStoreFetchRepository:
        """Create a FeatureStoreFetchRepository for testing."""
        mock_feature_store = Mock(spec=FeatureStore)
        mock_feast_provider.get_feature_store.return_value = mock_feature_store

        # Set up default mock responses to return actual DataFrames
        mock_offline_response = Mock()
        mock_offline_response.to_df.return_value = DataFrame({
            "event_timestamp": [pd.Timestamp("2024-01-01 09:00:00")],
            "symbol": ["EURUSD"],
            "rsi_14": [45.2]
        })
        mock_feature_store.get_historical_features.return_value = mock_offline_response

        return FeatureStoreFetchRepository(feast_provider=mock_feast_provider)

    @pytest.fixture
    def sample_timestamps(self) -> pd.Series:
        """Create sample timestamps for testing (UTC)."""
        return pd.Series([
            pd.Timestamp("2024-01-01 09:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 11:00:00", tz="UTC")
        ])

    def test_get_offline_first_call_creates_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline gets feature service on first call."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service

        # When
        result = repository.get_offline(
            feature_service_request=feature_service_request,
            timestamps=sample_timestamps
        )

        # Then
        mock_feast_provider.get_feature_service.assert_called_once()

        # Verify entity_df structure
        call_args = repository._fs.get_historical_features.call_args
        assert call_args[1]["features"] == mock_feature_service
        entity_df = call_args[1]["entity_df"]

        pd.testing.assert_series_equal(entity_df["event_timestamp"], sample_timestamps, check_names=False)
        assert all(entity_df["symbol"] == feature_service_request.symbol)

        assert isinstance(result, DataFrame)

    def test_get_offline_reuses_existing_feature_service(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline reuses existing feature service on subsequent calls."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service

        # When
        result = repository.get_offline(
            feature_service_request=feature_service_request,
            timestamps=sample_timestamps
        )

        # Then
        mock_feast_provider.get_feature_service.assert_called_once()
        repository._fs.get_historical_features.assert_called_once()
        assert isinstance(result, DataFrame)

    def test_get_offline_raises_error_when_feature_service_is_none(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline raises error when feature service creation returns None."""
        # Given
        mock_feast_provider.get_feature_service.side_effect = ValueError("Feature service not found")

        # When & Then
        with pytest.raises(ValueError, match="Feature service not found"):
            repository.get_offline(
                feature_service_request=feature_service_request,
                timestamps=sample_timestamps
            )

    def test_get_offline_with_different_symbol(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        gbpusd_feature_service_request: FeatureServiceRequestContainer,
        sample_timestamps: pd.Series
    ) -> None:
        """Test get_offline with different symbol."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service

        # When
        result = repository.get_offline(
            feature_service_request=gbpusd_feature_service_request,
            timestamps=sample_timestamps
        )

        # Then
        call_args = repository._fs.get_historical_features.call_args
        entity_df = call_args[1]["entity_df"]

        assert all(entity_df["symbol"] == gbpusd_feature_service_request.symbol)
        assert isinstance(result, DataFrame)

    def test_get_offline_with_empty_timestamps(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test get_offline with empty timestamps series."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        empty_timestamps = pd.Series([], dtype='datetime64[ns]')

        # When
        result = repository.get_offline(
            feature_service_request=feature_service_request,
            timestamps=empty_timestamps
        )

        # Then
        # Should return empty DataFrame without calling get_historical_features
        assert isinstance(result, DataFrame)
        assert len(result) == 0
        repository._fs.get_historical_features.assert_not_called()

    def test_get_offline_returns_dataframe_from_feast_response(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        mock_feature_service: Mock,
        feature_service_request: FeatureServiceRequestContainer,
        sample_timestamps: pd.Series
    ) -> None:
        """Test that get_offline returns the DataFrame from Feast response."""
        # Given
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_df = DataFrame({
            "event_timestamp": sample_timestamps,
            "symbol": [feature_service_request.symbol] * 3,
            "rsi_14": [30.5, 45.2, 67.8],
            "sma_20": [1.0850, 1.0855, 1.0860]
        })
        mock_response = Mock()
        mock_response.to_df.return_value = expected_df
        repository._fs.get_historical_features.return_value = mock_response

        # When
        result = repository.get_offline(
            feature_service_request=feature_service_request,
            timestamps=sample_timestamps
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
        mock_feast_provider.get_feature_store.return_value = mock_feature_store
        return FeatureStoreFetchRepository(feast_provider=mock_feast_provider)

    def test_get_online_handles_feast_provider_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online properly propagates FeastProvider exceptions."""
        # Given
        mock_feast_provider.get_feature_service.side_effect = Exception("Feast provider error")

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            repository.get_online(
                feature_service_request=feature_service_request
            )

    def test_get_offline_handles_feast_provider_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_offline properly propagates FeastProvider exceptions."""
        # Given
        mock_feast_provider.get_feature_service.side_effect = Exception("Feast provider error")
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])

        # When & Then
        with pytest.raises(Exception, match="Feast provider error"):
            repository.get_offline(
                feature_service_request=feature_service_request,
                timestamps=sample_timestamps
            )

    def test_get_online_handles_feature_store_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_online properly propagates FeatureStore exceptions."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        repository._fs.get_online_features.side_effect = Exception("Feature store error")

        # When & Then
        with pytest.raises(Exception, match="Feature store error"):
            repository.get_online(
                feature_service_request=feature_service_request
            )

    def test_get_offline_handles_feature_store_exception(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that get_offline properly propagates FeatureStore exceptions."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        repository._fs.get_historical_features.side_effect = Exception("Feature store error")
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])

        # When & Then
        with pytest.raises(Exception, match="Feature store error"):
            repository.get_offline(
                feature_service_request=feature_service_request,
                timestamps=sample_timestamps
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
        mock_feast_provider.get_feature_store.return_value = mock_feature_store
        return FeatureStoreFetchRepository(feast_provider=mock_feast_provider)

    def test_feature_service_created_once_for_multiple_online_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that feature service is retrieved for each online call with correct parameters."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        expected_service_name = f"observation_space_{feature_service_request.symbol}_{feature_service_request.timeframe.value}_{feature_service_request.feature_version_info.semver}_{feature_service_request.feature_version_info.hash}"

        # When
        repository.get_online(feature_service_request=feature_service_request)
        repository.get_online(feature_service_request=feature_service_request)
        repository.get_online(feature_service_request=feature_service_request)

        # Then
        assert mock_feast_provider.get_feature_service.call_count == 3
        mock_feast_provider.get_feature_service.assert_called_with(service_name=expected_service_name)

    def test_feature_service_created_once_for_multiple_offline_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that feature service is retrieved for each offline call with correct parameters."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        expected_service_name = f"observation_space_{feature_service_request.symbol}_{feature_service_request.timeframe.value}_{feature_service_request.feature_version_info.semver}_{feature_service_request.feature_version_info.hash}"

        # When
        repository.get_offline(feature_service_request=feature_service_request, timestamps=sample_timestamps)
        repository.get_offline(feature_service_request=feature_service_request, timestamps=sample_timestamps)
        repository.get_offline(feature_service_request=feature_service_request, timestamps=sample_timestamps)

        # Then
        assert mock_feast_provider.get_feature_service.call_count == 3
        mock_feast_provider.get_feature_service.assert_called_with(service_name=expected_service_name)

    def test_feature_service_shared_between_online_and_offline_calls(
        self,
        repository: FeatureStoreFetchRepository,
        mock_feast_provider: Mock,
        feature_service_request: FeatureServiceRequestContainer
    ) -> None:
        """Test that feature service is retrieved consistently for both online and offline calls."""
        # Given
        mock_feature_service = Mock(spec=FeatureService)
        mock_feast_provider.get_feature_service.return_value = mock_feature_service
        sample_timestamps = pd.Series([pd.Timestamp("2024-01-01 09:00:00")])
        expected_service_name = f"observation_space_{feature_service_request.symbol}_{feature_service_request.timeframe.value}_{feature_service_request.feature_version_info.semver}_{feature_service_request.feature_version_info.hash}"

        # When
        repository.get_online(feature_service_request=feature_service_request)
        repository.get_offline(feature_service_request=feature_service_request, timestamps=sample_timestamps)

        # Then
        assert mock_feast_provider.get_feature_service.call_count == 2
        mock_feast_provider.get_feature_service.assert_called_with(service_name=expected_service_name)
