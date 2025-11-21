"""
Reworked unit tests for FeastProvider class.

This test suite focuses on:
- Testing public interfaces only
- Using test data builders for maintainable test data
- Consolidated mock setup through fixtures
- Proper type annotations
- Behavior-driven testing rather than implementation details
"""

import logging
from unittest.mock import Mock, patch

import pytest
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32

from drl_trading_adapter.adapter.feature_store.offline import IOfflineFeatureRepository
from drl_trading_adapter.adapter.feature_store.provider.feast_provider import FeastProvider
from drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper import FeatureStoreWrapper
from drl_trading_adapter.adapter.feature_store.provider.mapper.feature_field_mapper import IFeatureFieldMapper
from drl_trading_common.base import BaseFeature
from drl_trading_common.config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model import FeatureViewRequestContainer


# ================================================================================================
# Test Data Builders
# ================================================================================================

class FeatureStoreConfigBuilder:
    """Builder for FeatureStoreConfig test data."""

    def __init__(self) -> None:
        self._ttl_days = 30
        self._entity_name = "default_entity"
        self._cache_enabled = True
        self._online_enabled = True
        self._service_name = "default_service"
        self._service_version = "1.0.0"

    def with_ttl_days(self, ttl_days: int) -> "FeatureStoreConfigBuilder":
        self._ttl_days = ttl_days
        return self

    def with_entity_name(self, entity_name: str) -> "FeatureStoreConfigBuilder":
        self._entity_name = entity_name
        return self

    def with_cache_disabled(self) -> "FeatureStoreConfigBuilder":
        self._cache_enabled = False
        return self

    def with_online_disabled(self) -> "FeatureStoreConfigBuilder":
        self._online_enabled = False
        return self

    def with_service_info(self, name: str, version: str) -> "FeatureStoreConfigBuilder":
        self._service_name = name
        self._service_version = version
        return self

    def build(self) -> Mock:
        """Build a mock FeatureStoreConfig with configured values."""
        config = Mock(spec=FeatureStoreConfig)
        config.ttl_days = self._ttl_days
        config.entity_name = self._entity_name
        config.cache_enabled = self._cache_enabled
        config.online_enabled = self._online_enabled
        config.service_name = self._service_name
        config.service_version = self._service_version
        return config


class FeatureRequestBuilder:
    """Builder for FeatureViewRequestContainer test data."""

    def __init__(self) -> None:
        self._symbol = "EURUSD"
        self._feature_role = FeatureRoleEnum.OBSERVATION_SPACE
        self._feature = Mock(spec=BaseFeature)
        self._feature.get_sub_features_names.return_value = ["sub1", "sub2"]
        self._timeframe = Timeframe.HOUR_1

    def with_symbol(self, symbol: str) -> "FeatureRequestBuilder":
        self._symbol = symbol
        return self

    def with_role(self, role: FeatureRoleEnum) -> "FeatureRequestBuilder":
        self._feature_role = role
        return self

    def with_feature(self, feature: Mock) -> "FeatureRequestBuilder":
        self._feature = feature
        return self

    def with_sub_features(self, sub_features: list[str]) -> "FeatureRequestBuilder":
        self._feature.get_sub_features_names.return_value = sub_features
        return self

    def with_timeframe(self, timeframe: Timeframe) -> "FeatureRequestBuilder":
        self._timeframe = timeframe
        return self

    def build(self) -> FeatureViewRequestContainer:
        """Build a FeatureViewRequestContainer with configured values."""
        return FeatureViewRequestContainer(
            symbol=self._symbol,
            feature_role=self._feature_role,
            feature=self._feature,
            timeframe=self._timeframe
        )


class FeastObjectBuilder:
    """Builder for Feast SDK objects (FeatureView, Entity, etc.)."""

    @staticmethod
    def create_feature_view(name: str = "test_feature_view") -> Mock:
        """Create a mock FeatureView."""
        feature_view = Mock(spec=FeatureView)
        feature_view.name = name
        return feature_view

    @staticmethod
    def create_entity(name: str = "test_entity") -> Mock:
        """Create a mock Entity."""
        entity = Mock(spec=Entity)
        entity.name = name
        return entity

    @staticmethod
    def create_feature_service(name: str = "test_service") -> Mock:
        """Create a mock FeatureService."""
        service = Mock(spec=FeatureService)
        service.name = name
        return service

    @staticmethod
    def create_field(name: str = "test_field") -> Field:
        """Create a real Field object."""
        return Field(name=name, dtype=Float32)

    @staticmethod
    def create_file_source(path: str = "/test/path.parquet") -> Mock:
        """Create a mock FileSource."""
        source = Mock(spec=FileSource)
        source.path = path
        return source


# ================================================================================================
# Test Constants
# ================================================================================================

class TestConstants:
    """Constants used across tests."""

    DEFAULT_SYMBOL = "EURUSD"
    ALTERNATIVE_SYMBOL = "GBPUSD"
    DEFAULT_SERVICE_NAME = "test_service"
    DEFAULT_FEATURE_NAME = "test_feature"
    DEFAULT_ENTITY_NAME = "test_entity"
    DEFAULT_TTL_DAYS = 30
    DEFAULT_REPO_PATH = "/test/data/EURUSD.parquet"


# ================================================================================================
# Test Fixtures
# ================================================================================================

class TestFeastProviderSetup:
    """Consolidated test setup with reusable fixtures."""

    @pytest.fixture
    def feature_store_config(self) -> Mock:
        """Standard feature store config for most tests."""
        return FeatureStoreConfigBuilder().build()

    @pytest.fixture
    def feature_store_config_cache_disabled(self) -> Mock:
        """Feature store config with cache disabled."""
        return FeatureStoreConfigBuilder().with_cache_disabled().build()

    @pytest.fixture
    def feature_store_config_online_disabled(self) -> Mock:
        """Feature store config with online mode disabled."""
        return FeatureStoreConfigBuilder().with_online_disabled().build()

    @pytest.fixture
    def mock_feature_store(self) -> Mock:
        """Mock FeatureStore with standard behavior."""
        store = Mock(spec=FeatureStore)
        store.apply.return_value = None
        store.list_feature_views.return_value = []
        # Note: Don't set default side_effect for get_entity and get_feature_service
        # Individual tests will configure these as needed
        return store

    @pytest.fixture
    def mock_feature_store_wrapper(self, mock_feature_store: Mock) -> Mock:
        """Mock FeatureStoreWrapper that returns the mock feature store."""
        wrapper = Mock(spec=FeatureStoreWrapper)
        wrapper.get_feature_store.return_value = mock_feature_store
        return wrapper

    @pytest.fixture
    def mock_offline_repo(self) -> Mock:
        """Mock offline repository with standard behavior."""
        repo = Mock(spec=IOfflineFeatureRepository)
        repo.get_repo_path.return_value = TestConstants.DEFAULT_REPO_PATH
        return repo

    @pytest.fixture
    def mock_feature_field_mapper(self) -> Mock:
        """Mock feature field mapper with standard behavior."""
        mapper = Mock(spec=IFeatureFieldMapper)
        mapper.get_field_base_name.return_value = TestConstants.DEFAULT_FEATURE_NAME
        mapper.create_fields.return_value = [FeastObjectBuilder.create_field()]
        return mapper

    @pytest.fixture
    def feast_provider(
        self,
        feature_store_config: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_repo: Mock,
        mock_feature_field_mapper: Mock,
    ) -> FeastProvider:
        """Standard FeastProvider instance for testing."""
        return FeastProvider(
            feature_store_config=feature_store_config,
            feature_store_wrapper=mock_feature_store_wrapper,
            offline_feature_repo=mock_offline_repo,
            feature_field_mapper=mock_feature_field_mapper,
        )

    @pytest.fixture
    def sample_feature_request(self) -> FeatureViewRequestContainer:
        """Standard feature request for testing."""
        return FeatureRequestBuilder().build()

    @pytest.fixture
    def multiple_feature_requests(self) -> list[FeatureViewRequestContainer]:
        """Multiple feature requests for batch testing."""
        return [
            FeatureRequestBuilder().with_symbol(TestConstants.DEFAULT_SYMBOL).build(),
            FeatureRequestBuilder().with_symbol(TestConstants.ALTERNATIVE_SYMBOL).build(),
        ]


# ================================================================================================
# Public Interface Tests
# ================================================================================================

class TestFeastProviderPublicInterface(TestFeastProviderSetup):
    """Test the public interface of FeastProvider."""

    def test_get_feature_store_returns_configured_store(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test that get_feature_store returns the configured FeatureStore instance."""
        # Given
        # FeastProvider is configured with a mock feature store

        # When
        result = feast_provider.get_feature_store()

        # Then
        assert result is mock_feature_store

    def test_get_offline_repo_returns_configured_repository(
        self,
        feast_provider: FeastProvider,
        mock_offline_repo: Mock
    ) -> None:
        """Test that get_offline_repo returns the configured offline repository."""
        # Given
        # FeastProvider is configured with a mock offline repository

        # When
        result = feast_provider.get_offline_repo()

        # Then
        assert result is mock_offline_repo


class TestFeastProviderFeatureServiceManagement(TestFeastProviderSetup):
    """Test feature service management through public interface."""

    def test_get_feature_service_returns_existing_service(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test retrieving an existing feature service."""
        # Given
        expected_service = FeastObjectBuilder.create_feature_service(TestConstants.DEFAULT_SERVICE_NAME)
        mock_feature_store.get_feature_service.return_value = expected_service

        # When
        result = feast_provider.get_feature_service(TestConstants.DEFAULT_SERVICE_NAME)

        # Then
        assert result is expected_service
        mock_feature_store.get_feature_service.assert_called_once_with(
            name=TestConstants.DEFAULT_SERVICE_NAME,
            allow_cache=True
        )

    def test_get_feature_service_raises_error_when_not_found(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test that get_feature_service raises RuntimeError when service doesn't exist."""
        # Given
        mock_feature_store.get_feature_service.side_effect = Exception("Service not found")

        # When / Then
        with pytest.raises(RuntimeError, match="Failed to get feature service 'nonexistent_service'"):
            feast_provider.get_feature_service("nonexistent_service")

    def test_get_feature_service_respects_cache_configuration(
        self,
        mock_feature_store_wrapper: Mock,
        mock_offline_repo: Mock,
        mock_feature_field_mapper: Mock,
        feature_store_config_cache_disabled: Mock,
        mock_feature_store: Mock
    ) -> None:
        """Test that get_feature_service respects cache configuration."""
        # Given
        feast_provider = FeastProvider(
            feature_store_config=feature_store_config_cache_disabled,
            feature_store_wrapper=mock_feature_store_wrapper,
            offline_feature_repo=mock_offline_repo,
            feature_field_mapper=mock_feature_field_mapper,
        )
        expected_service = FeastObjectBuilder.create_feature_service()
        mock_feature_store.get_feature_service.return_value = expected_service

        # When
        feast_provider.get_feature_service(TestConstants.DEFAULT_SERVICE_NAME)

        # Then
        mock_feature_store.get_feature_service.assert_called_once_with(
            name=TestConstants.DEFAULT_SERVICE_NAME,
            allow_cache=False
        )


class TestFeastProviderFeatureServiceCreation(TestFeastProviderSetup):
    """Test feature service creation through public interface."""

    def test_get_or_create_feature_service_returns_existing_service(
        self,
        feast_provider: FeastProvider,
        sample_feature_request: FeatureViewRequestContainer,
        mock_feature_store: Mock
    ) -> None:
        """Test that get_or_create_feature_service returns existing service when available."""
        # Given
        existing_service = FeastObjectBuilder.create_feature_service(TestConstants.DEFAULT_SERVICE_NAME)
        mock_feature_store.get_feature_service.return_value = existing_service

        # When
        result = feast_provider.get_or_create_feature_service(
            TestConstants.DEFAULT_SERVICE_NAME,
            [sample_feature_request]
        )

        # Then
        assert result.name == existing_service.name  # Compare names instead of object identity
        # Should not call apply since service already exists
        mock_feature_store.apply.assert_not_called()

    @patch('drl_trading_adapter.adapter.feature_store.provider.feast_provider.FeatureService')
    def test_get_or_create_feature_service_creates_new_service(
        self,
        mock_feature_service_class: Mock,
        feast_provider: FeastProvider,
        sample_feature_request: FeatureViewRequestContainer,
        mock_feature_store: Mock,
        mock_feature_field_mapper: Mock,
        mock_offline_repo: Mock
    ) -> None:
        """Test creating a new feature service when it doesn't exist."""
        # Given
        mock_feature_store.get_feature_service.side_effect = Exception("Not found")
        new_service = FeastObjectBuilder.create_feature_service(TestConstants.DEFAULT_SERVICE_NAME)
        mock_feature_service_class.return_value = new_service

        # When
        result = feast_provider.get_or_create_feature_service(
            TestConstants.DEFAULT_SERVICE_NAME,
            [sample_feature_request]
        )

        # Then
        assert result is new_service
        # Verify the service was created and applied (could be called multiple times during feature view creation)
        assert mock_feature_store.apply.call_count >= 1
        # Verify the service was created with correct parameters
        mock_feature_service_class.assert_called_once()
        call_kwargs = mock_feature_service_class.call_args[1]
        assert call_kwargs['name'] == TestConstants.DEFAULT_SERVICE_NAME
        assert 'features' in call_kwargs

    def test_get_or_create_feature_service_with_empty_requests_raises_error(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test that creating service with empty requests raises ValueError."""
        # Given
        mock_feature_store.get_feature_service.side_effect = Exception("Not found")

        # When / Then
        with pytest.raises(ValueError, match="No feature views available"):
            feast_provider.get_or_create_feature_service(TestConstants.DEFAULT_SERVICE_NAME, [])

    def test_get_or_create_feature_service_logs_creation(
        self,
        feast_provider: FeastProvider,
        sample_feature_request: FeatureViewRequestContainer,
        mock_feature_store: Mock,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that feature service creation is properly logged."""
        # Given
        mock_feature_store.get_feature_service.side_effect = Exception("Not found")
        caplog.set_level(logging.INFO)

        with patch('drl_trading_adapter.adapter.feature_store.provider.feast_provider.FeatureService'):
            # When
            feast_provider.get_or_create_feature_service(
                TestConstants.DEFAULT_SERVICE_NAME,
                [sample_feature_request]
            )

            # Then
            assert f"Creating feature service '{TestConstants.DEFAULT_SERVICE_NAME}'" in caplog.text


class TestFeastProviderEdgeCases(TestFeastProviderSetup):
    """Test edge cases and error scenarios."""

    def test_handles_special_characters_in_service_name(
        self,
        feast_provider: FeastProvider,
        sample_feature_request: FeatureViewRequestContainer,
        mock_feature_store: Mock
    ) -> None:
        """Test handling of service names with special characters."""
        # Given
        special_service_name = "service-with-special_chars.v1"
        existing_service = FeastObjectBuilder.create_feature_service(special_service_name)
        mock_feature_store.get_feature_service.return_value = existing_service

        # When
        result = feast_provider.get_feature_service(special_service_name)

        # Then
        assert result is existing_service

    def test_handles_unicode_in_service_name(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test handling of Unicode characters in service names."""
        # Given
        unicode_service_name = "service_with_unicode_字符"
        existing_service = FeastObjectBuilder.create_feature_service(unicode_service_name)
        mock_feature_store.get_feature_service.return_value = existing_service

        # When
        result = feast_provider.get_feature_service(unicode_service_name)

        # Then
        assert result is existing_service

    def test_handles_very_long_service_name(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock
    ) -> None:
        """Test handling of extremely long service names."""
        # Given
        long_service_name = "very_long_service_name_" * 50  # Very long name
        existing_service = FeastObjectBuilder.create_feature_service(long_service_name)
        mock_feature_store.get_feature_service.return_value = existing_service

        # When
        result = feast_provider.get_feature_service(long_service_name)

        # Then
        assert result is existing_service


class TestFeastProviderConfigurationVariants(TestFeastProviderSetup):
    """Test behavior with different configuration variants."""

    def test_configuration_with_different_ttl_values(
        self,
        mock_feature_store_wrapper: Mock,
        mock_offline_repo: Mock,
        mock_feature_field_mapper: Mock
    ) -> None:
        """Test FeastProvider with various TTL configurations."""
        # Given
        ttl_values = [1, 7, 30, 365]

        for ttl_days in ttl_values:
            # When
            config = FeatureStoreConfigBuilder().with_ttl_days(ttl_days).build()
            provider = FeastProvider(
                feature_store_config=config,
                feature_store_wrapper=mock_feature_store_wrapper,
                offline_feature_repo=mock_offline_repo,
                feature_field_mapper=mock_feature_field_mapper,
            )

            # Then
            assert provider.feature_store_config.ttl_days == ttl_days

    def test_configuration_with_different_entity_names(
        self,
        mock_feature_store_wrapper: Mock,
        mock_offline_repo: Mock,
        mock_feature_field_mapper: Mock
    ) -> None:
        """Test FeastProvider with various entity name configurations."""
        # Given
        entity_names = ["default_entity", "custom_entity", "trading_entity"]

        for entity_name in entity_names:
            # When
            config = FeatureStoreConfigBuilder().with_entity_name(entity_name).build()
            provider = FeastProvider(
                feature_store_config=config,
                feature_store_wrapper=mock_feature_store_wrapper,
                offline_feature_repo=mock_offline_repo,
                feature_field_mapper=mock_feature_field_mapper,
            )

            # Then
            assert provider.feature_store_config.entity_name == entity_name


class TestFeastProviderIntegrationWorkflows(TestFeastProviderSetup):
    """Test complete workflows through public interface."""

    @patch('drl_trading_adapter.adapter.feature_store.provider.feast_provider.FeatureService')
    def test_complete_feature_service_creation_workflow(
        self,
        mock_feature_service_class: Mock,
        feast_provider: FeastProvider,
        multiple_feature_requests: list[FeatureViewRequestContainer],
        mock_feature_store: Mock
    ) -> None:
        """Test the complete workflow of creating a feature service with multiple requests."""
        # Given
        mock_feature_store.get_feature_service.side_effect = Exception("Not found")
        new_service = FeastObjectBuilder.create_feature_service(TestConstants.DEFAULT_SERVICE_NAME)
        mock_feature_service_class.return_value = new_service

        # When
        result = feast_provider.get_or_create_feature_service(
            TestConstants.DEFAULT_SERVICE_NAME,
            multiple_feature_requests
        )

        # Then
        assert result is new_service
        # Verify all components were called appropriately (may be called multiple times during creation)
        assert mock_feature_store.apply.call_count >= 1
        mock_feature_service_class.assert_called_once()

    def test_feature_service_retrieval_with_logging(
        self,
        feast_provider: FeastProvider,
        mock_feature_store: Mock,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that feature service operations generate appropriate logs."""
        # Given
        existing_service = FeastObjectBuilder.create_feature_service(TestConstants.DEFAULT_SERVICE_NAME)
        mock_feature_store.get_feature_service.return_value = existing_service
        caplog.set_level(logging.DEBUG)

        # When
        feast_provider.get_feature_service(TestConstants.DEFAULT_SERVICE_NAME)

        # Then
        # Should not log anything for successful retrieval (no debug logs in current implementation)
        assert len(caplog.records) == 0


if __name__ == "__main__":
    pytest.main([__file__])
