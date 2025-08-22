import os
import tempfile
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import Mock

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.config.feature_config import FeatureStoreConfig
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from feast import Entity, FeatureService, FeatureStore, FeatureView
from feast.types import Float32

from drl_trading_core.preprocess.feature.feature_manager import FeatureManager
from drl_trading_core.preprocess.feature_store.port.offline_feature_repo_interface import (
    IOfflineFeatureRepository,
)
from drl_trading_adapter.adapter.feature_store.feast.feast_provider import (
    FeastProvider,
)
from drl_trading_adapter.adapter.feature_store.feast.feature_store_wrapper import (
    FeatureStoreWrapper,
)


class TestFeastProvider:
    """Test cases for FeastProvider class."""

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def feature_store_config(self, temp_dir: str) -> FeatureStoreConfig:
        """Create a test feature store configuration."""
        return FeatureStoreConfig(
            enabled=True,
            config_directory=temp_dir,
            entity_name="trading_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

    @pytest.fixture
    def disabled_feature_store_config(self, temp_dir: str) -> FeatureStoreConfig:
        """Create a disabled feature store configuration."""
        return FeatureStoreConfig(
            enabled=False,
            config_directory=temp_dir,
            entity_name="trading_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

    @pytest.fixture
    def mock_feature_manager(self) -> Mock:
        """Create a mock feature manager."""
        mock_manager = Mock(spec=FeatureManager)
        return mock_manager

    @pytest.fixture
    def mock_feature_store_wrapper(self) -> Mock:
        """Create a mock feature store wrapper."""
        mock_wrapper = Mock(spec=FeatureStoreWrapper)
        mock_store_instance = Mock(spec=FeatureStore)
        mock_store_instance.repo_path = "/test/repo/path"
        # Configure get_feature_service to return None (service doesn't exist)
        mock_store_instance.get_feature_service.return_value = None
        mock_wrapper.get_feature_store.return_value = mock_store_instance
        return mock_wrapper

    @pytest.fixture
    def mock_offline_feature_repo(self, temp_dir: str) -> Mock:
        """Create a mock offline feature repository."""
        mock_repo = Mock(spec=IOfflineFeatureRepository)
        mock_repo.get_repo_path.return_value = os.path.join(temp_dir, "EURUSD")
        return mock_repo

    @pytest.fixture
    def mock_feature_config(self) -> Mock:
        """Create a mock feature configuration."""
        mock_config = Mock(spec=BaseParameterSetConfig)
        mock_config.hash_id.return_value = "test_hash_123"
        return mock_config

    @pytest.fixture
    def mock_observation_feature(self, mock_feature_config: Mock) -> Mock:
        """Create a mock observation space feature."""
        mock_feature = Mock(spec=BaseFeature)
        mock_feature.get_feature_name.return_value = "rsi_feature"
        mock_feature.get_config.return_value = mock_feature_config
        mock_feature.get_sub_features_names.return_value = ["rsi_14", "rsi_21"]
        mock_feature.get_feature_role.return_value = FeatureRoleEnum.OBSERVATION_SPACE
        mock_feature.get_config_to_string.return_value = "-"
        return mock_feature

    @pytest.fixture
    def mock_reward_feature(self, mock_feature_config: Mock) -> Mock:
        """Create a mock reward engineering feature."""
        mock_feature = Mock(spec=BaseFeature)
        mock_feature.get_feature_name.return_value = "profit_feature"
        mock_feature.get_config.return_value = mock_feature_config
        mock_feature.get_sub_features_names.return_value = ["profit_target", "stop_loss"]
        mock_feature.get_feature_role.return_value = FeatureRoleEnum.REWARD_ENGINEERING
        mock_feature.get_config_to_string.return_value = "-"
        return mock_feature

    @pytest.fixture
    def symbol(self) -> str:
        """Create a test symbol."""
        return "EURUSD"

    @pytest.fixture
    def feature_version_info(self) -> FeatureConfigVersionInfo:
        """Create a test feature version info."""
        return FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="abc123def456",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[],
            description="Test feature version"
        )

    def test_init_with_enabled_config(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock,
        temp_dir: str
    ) -> None:
        """Test FeastProvider initialization with enabled configuration."""
        # Given
        # Setup is done by fixtures

        # When
        provider = FeastProvider(
            feature_store_config=feature_store_config,
            feature_manager=mock_feature_manager,
            feature_store_wrapper=mock_feature_store_wrapper,
            offline_feature_repo=mock_offline_feature_repo
        )

        # Then
        assert provider.feature_manager == mock_feature_manager
        assert provider.feature_store_config == feature_store_config
        assert provider.feature_store == mock_feature_store_wrapper.get_feature_store.return_value
        assert provider._offline_repo == mock_offline_feature_repo
        mock_feature_store_wrapper.get_feature_store.assert_called_once()

    def test_init_with_disabled_config(
        self,
        disabled_feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test FeastProvider initialization with disabled configuration."""
        # Given
        # Setup is done by fixtures

        # When
        provider = FeastProvider(
            feature_store_config=disabled_feature_store_config,
            feature_manager=mock_feature_manager,
            feature_store_wrapper=mock_feature_store_wrapper,
            offline_feature_repo=mock_offline_feature_repo
        )

        # Then
        assert provider.feature_manager == mock_feature_manager
        assert provider.feature_store_config == disabled_feature_store_config
        assert provider.feature_store == mock_feature_store_wrapper.get_feature_store.return_value
        assert provider._offline_repo == mock_offline_feature_repo
        mock_feature_store_wrapper.get_feature_store.assert_called_once()

    def test_get_feature_store(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test getting the feature store instance."""
        # Given
        provider = FeastProvider(
            feature_store_config,
            mock_feature_manager,
            mock_feature_store_wrapper,
            mock_offline_feature_repo
        )

        # When
        result = provider.get_feature_store()

        # Then
        assert result == mock_feature_store_wrapper.get_feature_store.return_value

    def test_is_enabled_returns_true(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock,
    ) -> None:
        """Test is_enabled returns True for enabled configuration."""
        # Given
        provider = FeastProvider(
            feature_store_config,
            mock_feature_manager,
            mock_feature_store_wrapper,
            mock_offline_feature_repo
        )

        # When
        result = provider.is_enabled()

        # Then
        assert result is True

    def test_is_enabled_returns_false(
        self,
        disabled_feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test is_enabled returns False for disabled configuration."""
        # Given
        provider = FeastProvider(
            disabled_feature_store_config,
            mock_feature_manager,
            mock_feature_store_wrapper,
            mock_offline_feature_repo
        )

        # When
        result = provider.is_enabled()

        # Then
        assert result is False

    def test_resolve_feature_store_path_absolute_path(
        self,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        temp_dir: str,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test resolving absolute feature store path."""
        # Given
        absolute_path = os.path.abspath(temp_dir)
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=absolute_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        # When
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # Then
        # The wrapper should be called to get the feature store
        mock_feature_store_wrapper.get_feature_store.assert_called_once()
        assert provider.feature_store == mock_feature_store_wrapper.get_feature_store.return_value

    def test_resolve_feature_store_path_relative_path(
        self,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test resolving relative feature store path."""
        # Given
        relative_path = "relative/path/to/store"
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=relative_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        # When
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # Then
        # The wrapper should be called to get the feature store
        mock_feature_store_wrapper.get_feature_store.assert_called_once()
        assert provider.feature_store == mock_feature_store_wrapper.get_feature_store.return_value

    def test_create_fields_single_feature(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_observation_feature: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test creating fields for a single feature."""
        # Given
        provider = FeastProvider(feature_store_config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # When
        fields = provider._create_fields(mock_observation_feature)

        # Then
        assert len(fields) == 2
        assert fields[0].name == "rsi_feature_-_test_hash_123_rsi_14"
        assert fields[0].dtype == Float32
        assert fields[1].name == "rsi_feature_-_test_hash_123_rsi_21"
        assert fields[1].dtype == Float32

    def test_create_feature_view_observation_space(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_observation_feature: Mock,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        mock_offline_feature_repo: Mock,
    ) -> None:
        """Test creating feature view for observation space features."""
        # Given
        provider = FeastProvider(feature_store_config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)
        mock_feature_manager.get_features_by_role.return_value = [mock_observation_feature]
        feature_view_name = "test_observation_view"

        # When
        feature_view = provider.create_feature_view(
            symbol=symbol,
            feature_view_name=feature_view_name,
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info
        )

        # Then
        assert isinstance(feature_view, FeatureView)
        assert feature_view.name == feature_view_name
        assert len(feature_view.schema) == 2  # Two sub-features
        assert feature_view.ttl == timedelta(days=30)
        assert not feature_view.online
        assert feature_view.tags["symbol"] == "EURUSD"
        mock_feature_manager.get_features_by_role.assert_called_once_with(FeatureRoleEnum.OBSERVATION_SPACE)

    def test_create_feature_service(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        symbol: str,
        feature_version_info: FeatureConfigVersionInfo,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test creating feature service."""
        # Given
        provider = FeastProvider(feature_store_config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)
        mock_feature_views = []
        for _ in range(2):
            mock_fv = Mock(spec=FeatureView)
            mock_fv.projection = Mock()  # Add required projection attribute
            mock_feature_views.append(mock_fv)

        # When
        feature_service = provider.create_feature_service(
            feature_views=mock_feature_views,
            symbol=symbol,
            feature_version_info=feature_version_info
        )

        # Then
        assert isinstance(feature_service, FeatureService)
        expected_name = f"service_{symbol}_v{feature_version_info.semver}-{feature_version_info.hash}"
        assert feature_service.name == expected_name
        assert len(feature_service._features) == 2  # Check internal _features attribute

    def test_get_entity(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        symbol: str,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test getting entity for symbol."""
        # Given
        provider = FeastProvider(feature_store_config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # When
        entity = provider.get_entity(symbol)

        # Then
        assert isinstance(entity, Entity)
        assert entity.name == "trading_entity"
        assert entity.join_key == "symbol"  # Single join key
        assert "EURUSD" in entity.description

    def test_get_feature_name(
        self,
        feature_store_config: FeatureStoreConfig,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_feature_config: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test getting feature name with config hash."""
        # Given
        provider = FeastProvider(feature_store_config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)
        mock_feature = Mock(spec=BaseFeature)
        mock_config = Mock(spec=BaseParameterSetConfig)
        mock_config.hash_id.return_value = "a1b2c3"
        mock_feature.get_feature_name.return_value = "feature"
        mock_feature.get_config.return_value = mock_config
        mock_feature.get_config_to_string.return_value = "12_2"

        # When
        result = provider._get_field_base_name(mock_feature)

        # Then
        assert result == "feature_12_2_a1b2c3"
        mock_feature.get_config.assert_called_once()
        mock_feature.get_config_to_string.assert_called_once()
        mock_config.hash_id.assert_called_once()


class TestFeastProviderParametrized:
    """Parametrized tests for FeastProvider class."""

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_feature_manager(self) -> Mock:
        """Create a mock feature manager."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def mock_feature_store_wrapper(self) -> Mock:
        """Create a mock feature store wrapper."""
        mock_wrapper = Mock(spec=FeatureStoreWrapper)
        mock_store_instance = Mock(spec=FeatureStore)
        mock_store_instance.repo_path = "/test/repo/path"
        # Configure get_feature_service to return None (service doesn't exist)
        mock_store_instance.get_feature_service.return_value = None
        mock_wrapper.get_feature_store.return_value = mock_store_instance
        return mock_wrapper

    @pytest.fixture
    def mock_offline_feature_repo(self, temp_dir: str) -> Mock:
        """Create a mock offline feature repository."""
        mock_repo = Mock(spec=IOfflineFeatureRepository)
        mock_repo.get_repo_path.return_value = os.path.join(temp_dir, "EURUSD")
        return mock_repo

    @pytest.mark.parametrize(
        "enabled,expected_result",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_is_enabled_parametrized(
        self,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        temp_dir: str,
        enabled: bool,
        expected_result: bool,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test is_enabled method with different configuration states."""
        # Given
        config = FeatureStoreConfig(
            enabled=enabled,
            config_directory=temp_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # When
        result = provider.is_enabled()

        # Then
        assert result == expected_result

    @pytest.mark.parametrize(
        "feature_role,features_count,expected_fields_count",
        [
            (FeatureRoleEnum.OBSERVATION_SPACE, 1, 2),  # One feature with 2 sub-features
            (FeatureRoleEnum.REWARD_ENGINEERING, 1, 2),  # One feature with 2 sub-features
            (FeatureRoleEnum.OBSERVATION_SPACE, 2, 4),  # Two features with 2 sub-features each
        ],
    )
    def test_create_feature_view_with_different_feature_counts(
        self,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        temp_dir: str,
        feature_role: FeatureRoleEnum,
        features_count: int,
        expected_fields_count: int,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test creating feature views with different numbers of features."""
        # Given
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=temp_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # Create mock features
        mock_features = []
        for i in range(features_count):
            mock_feature = Mock(spec=BaseFeature)
            mock_config = Mock(spec=BaseParameterSetConfig)
            mock_config.hash_id.return_value = f"hash_{i}"
            mock_feature.get_feature_name.return_value = f"feature_{i}"
            mock_feature.get_config.return_value = mock_config
            mock_feature.get_sub_features_names.return_value = ["sub_1", "sub_2"]
            mock_feature.get_config_to_string.return_value = "-"
            mock_features.append(mock_feature)

        mock_feature_manager.get_features_by_role.return_value = mock_features

        symbol = "EURUSD"
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When
        feature_view = provider.create_feature_view(
            symbol=symbol,
            feature_view_name="test_view",
            feature_role=feature_role,
            feature_version_info=feature_version_info
        )

        # Then
        assert len(feature_view.schema) == expected_fields_count
        mock_feature_manager.get_features_by_role.assert_called_once_with(feature_role)


class TestFeastProviderEdgeCases:
    """Test edge cases and error conditions for FeastProvider."""

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_feature_manager(self) -> Mock:
        """Create a mock feature manager."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def mock_feature_store_wrapper(self) -> Mock:
        """Create a mock feature store wrapper."""
        mock_wrapper = Mock(spec=FeatureStoreWrapper)
        mock_store_instance = Mock(spec=FeatureStore)
        mock_store_instance.repo_path = "/test/repo/path"
        # Configure get_feature_service to return None (service doesn't exist)
        mock_store_instance.get_feature_service.return_value = None
        mock_wrapper.get_feature_store.return_value = mock_store_instance
        return mock_wrapper

    @pytest.fixture
    def mock_offline_feature_repo(self, temp_dir: str) -> Mock:
        """Create a mock offline feature repository."""
        mock_repo = Mock(spec=IOfflineFeatureRepository)
        mock_repo.get_repo_path.return_value = os.path.join(temp_dir, "EURUSD")
        return mock_repo

    def test_create_fields_empty_sub_features(
        self,
        temp_dir: str,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test creating fields when feature has no sub-features."""
        # Given
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=temp_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        mock_feature = Mock(spec=BaseFeature)
        mock_feature_config = Mock(spec=BaseParameterSetConfig)
        mock_feature_config.hash_id.return_value = "empty_hash"
        mock_feature.get_feature_name.return_value = "empty_feature"
        mock_feature.get_config.return_value = mock_feature_config
        mock_feature.get_sub_features_names.return_value = []  # Empty sub-features
        mock_feature.get_config_to_string.return_value = "10_2"

        # When
        fields = provider._create_fields(mock_feature)

        # Then
        assert len(fields) == 1

    def test_create_feature_view_no_features_for_role(
        self,
        temp_dir: str,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test creating feature view when no features exist for the role."""
        # Given
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=temp_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)
        mock_feature_manager.get_features_by_role.return_value = []  # No features

        symbol = "EURUSD"
        feature_version_info = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime.now(),
            feature_definitions=[]
        )

        # When
        feature_view = provider.create_feature_view(
            symbol=symbol,
            feature_view_name="empty_view",
            feature_role=FeatureRoleEnum.OBSERVATION_SPACE,
            feature_version_info=feature_version_info
        )

        # Then
        assert len(feature_view.schema) == 0
        mock_feature_manager.get_features_by_role.assert_called_once_with(FeatureRoleEnum.OBSERVATION_SPACE)

    def test_provider_with_very_long_paths(
        self,
        mock_feature_manager: Mock,
        mock_feature_store_wrapper: Mock,
        mock_offline_feature_repo: Mock
    ) -> None:
        """Test provider initialization with very long file paths."""
        # Given
        long_path = os.path.join("very", "long", "path", "to", "feature", "store") * 10
        config = FeatureStoreConfig(
            enabled=True,
            config_directory=long_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        # When
        provider = FeastProvider(config, mock_feature_manager, mock_feature_store_wrapper, mock_offline_feature_repo)

        # Then
        # Should handle long paths without errors (initialization should succeed)
        assert provider.feature_store_config.config_directory == long_path
        mock_feature_store_wrapper.get_feature_store.assert_called_once()
