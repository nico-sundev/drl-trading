"""Unit tests for FeatureStoreWrapper.

This module tests the FeatureStoreWrapper class functionality including:
- Absolute and relative path resolution
- Disabled config behavior
- FeatureStore instance caching
- Error handling for invalid configurations
"""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig  # type: ignore
from feast import FeatureStore

from drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper import (  # type: ignore
    FeatureStoreWrapper,
)


class TestFeatureStoreWrapper:
    """Test suite for FeatureStoreWrapper."""

    @pytest.fixture
    def enabled_config(self) -> FeatureStoreConfig:
        """Create an enabled feature store config for testing."""
        return FeatureStoreConfig(
            enabled=True,
            repo_path="/test/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

    @pytest.fixture
    def disabled_config(self) -> FeatureStoreConfig:
        """Create a disabled feature store config for testing."""
        return FeatureStoreConfig(
            enabled=False,
            repo_path="/test/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

    @pytest.fixture
    def temp_feature_store_dir(self) -> Generator[str, None, None]:
        """Create a temporary feature store directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_init_with_enabled_config(self, enabled_config: FeatureStoreConfig) -> None:
        """Test initialization with enabled feature store config."""
        wrapper = FeatureStoreWrapper(enabled_config)

        assert wrapper._feature_store_config == enabled_config
        assert wrapper._feature_store is None

    def test_init_with_disabled_config(self, disabled_config: FeatureStoreConfig) -> None:
        """Test initialization with disabled feature store config."""
        wrapper = FeatureStoreWrapper(disabled_config)

        assert wrapper._feature_store_config == disabled_config
        assert wrapper._feature_store is None

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_get_feature_store_with_enabled_config(
        self,
        mock_feast_store: MagicMock,
        temp_feature_store_dir: str
    ) -> None:
        """Test get_feature_store with enabled config."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config)
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        mock_feast_store.assert_called_once_with(repo_path=temp_feature_store_dir)

    def test_get_feature_store_with_disabled_config(self, disabled_config: FeatureStoreConfig) -> None:
        """Test get_feature_store returns None when feature store is disabled."""
        wrapper = FeatureStoreWrapper(disabled_config)

        # This should return None due to disabled config in _resolve_feature_store_path
        result = wrapper._resolve_feature_store_path()

        assert result is None

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_get_feature_store_with_absolute_path(
        self,
        mock_feast_store: MagicMock,
        temp_feature_store_dir: str
    ) -> None:
        """Test get_feature_store with absolute path to config file."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,  # This is already absolute
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config)
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        mock_feast_store.assert_called_once_with(repo_path=temp_feature_store_dir)

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_get_feature_store_with_relative_path(
        self,
        mock_feast_store: MagicMock
    ) -> None:
        """Test get_feature_store with relative path to config file."""
        relative_path = "test_config/feature_store"
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=relative_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config)
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        # Verify that the path was converted to absolute
        call_args = mock_feast_store.call_args_list[0]
        called_path = call_args[1]['repo_path']
        assert os.path.isabs(called_path)
        # Normalize both paths for comparison (handles mixed separators)
        assert os.path.normpath(called_path).endswith(os.path.normpath(relative_path))

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_get_feature_store_caching(
        self,
        mock_feast_store: MagicMock,
        temp_feature_store_dir: str
    ) -> None:
        """Test that FeatureStore instance is cached after first creation."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config)

        # First call should create the instance
        result1 = wrapper.get_feature_store()
        assert result1 == mock_feast_instance
        assert mock_feast_store.call_count == 1

        # Second call should return cached instance
        result2 = wrapper.get_feature_store()
        assert result2 == mock_feast_instance
        assert result1 is result2
        assert mock_feast_store.call_count == 1  # Should not be called again

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_get_feature_store_creation_error(
        self,
        mock_feast_store: MagicMock
    ) -> None:
        """Test error handling when FeatureStore creation fails."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path="/nonexistent/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_store.side_effect = Exception("Failed to create FeatureStore")

        wrapper = FeatureStoreWrapper(config)

        with pytest.raises(Exception, match="Failed to create FeatureStore"):
            wrapper.get_feature_store()

    def test_resolve_feature_store_path_disabled(self, disabled_config: FeatureStoreConfig) -> None:
        """Test _resolve_feature_store_path returns None when disabled."""
        wrapper = FeatureStoreWrapper(disabled_config)

        result = wrapper._resolve_feature_store_path()

        assert result is None

    def test_resolve_feature_store_path_absolute(self, temp_feature_store_dir: str) -> None:
        """Test _resolve_feature_store_path with absolute path."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        wrapper = FeatureStoreWrapper(config)
        result = wrapper._resolve_feature_store_path()

        assert result == temp_feature_store_dir
        assert os.path.isabs(result)

    def test_resolve_feature_store_path_relative(self) -> None:
        """Test _resolve_feature_store_path with relative path."""
        relative_path = "test_config/feature_store"
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=relative_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        wrapper = FeatureStoreWrapper(config)
        result = wrapper._resolve_feature_store_path()

        assert result is not None
        assert os.path.isabs(result)
        # Normalize both paths for comparison (handles mixed separators)
        assert os.path.normpath(result).endswith(os.path.normpath(relative_path))

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_multiple_wrappers_independent_caching(
        self,
        mock_feast_store: MagicMock,
        temp_feature_store_dir: str
    ) -> None:
        """Test that multiple wrapper instances have independent caching."""
        mock_feast_instance1 = MagicMock(spec=FeatureStore)
        mock_feast_instance2 = MagicMock(spec=FeatureStore)
        mock_feast_store.side_effect = [mock_feast_instance1, mock_feast_instance2]

        # Create two configs with the same path
        config1 = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        config2 = FeatureStoreConfig(
            enabled=True,
            repo_path=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        # Create two wrapper instances
        wrapper1 = FeatureStoreWrapper(config1)
        wrapper2 = FeatureStoreWrapper(config2)

        # Each should create its own FeatureStore instance
        result1 = wrapper1.get_feature_store()
        result2 = wrapper2.get_feature_store()

        assert result1 == mock_feast_instance1
        assert result2 == mock_feast_instance2
        assert result1 is not result2
        assert mock_feast_store.call_count == 2

    @patch('drl_trading_core.preprocess.feature_store.provider.feature_store_wrapper.FeatureStore')
    def test_path_with_special_characters(
        self,
        mock_feast_store: MagicMock
    ) -> None:
        """Test handling of paths with special characters."""
        special_path = "config/test-feature_store@v1"
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=special_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config)
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        call_args = mock_feast_store.call_args_list[0]
        called_path = call_args[1]['repo_path']
        assert os.path.isabs(called_path)
        # Normalize both paths for comparison (handles mixed separators)
        assert os.path.normpath(called_path).endswith(os.path.normpath(special_path))

    def test_config_object_attributes(self, enabled_config: FeatureStoreConfig) -> None:
        """Test that config object has expected attributes."""
        wrapper = FeatureStoreWrapper(enabled_config)

        assert wrapper._feature_store_config.enabled is True
        assert wrapper._feature_store_config.repo_path == "/test/path/feature_store"
        assert wrapper._feature_store_config.entity_name == "test_entity"
        assert wrapper._feature_store_config.ttl_days == 30
        assert wrapper._feature_store_config.online_enabled is False
        assert wrapper._feature_store_config.service_name == "test_service"
        assert wrapper._feature_store_config.service_version == "1.0.0"

    @pytest.mark.parametrize("enabled,expected_none", [
        (True, False),
        (False, True),
    ])
    def test_resolve_path_parametrized(self, enabled: bool, expected_none: bool) -> None:
        """Test path resolution with parametrized enabled/disabled states."""
        config = FeatureStoreConfig(
            enabled=enabled,
            repo_path="test/path",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        wrapper = FeatureStoreWrapper(config)
        result = wrapper._resolve_feature_store_path()

        if expected_none:
            assert result is None
        else:
            assert result is not None
            assert os.path.isabs(result)

    @pytest.mark.parametrize("repo_path,is_absolute", [
        ("/absolute/path/to/repo", True),
        ("relative/path/to/repo", False),
        ("C:\\Windows\\path\\to\\repo", True),  # Windows absolute path
        ("./relative/path", False),
        ("../parent/path", False),
    ])
    def test_path_resolution_types(self, repo_path: str, is_absolute: bool) -> None:
        """Test path resolution with various path types."""
        config = FeatureStoreConfig(
            enabled=True,
            repo_path=repo_path,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0"
        )

        wrapper = FeatureStoreWrapper(config)
        result = wrapper._resolve_feature_store_path()

        assert result is not None
        assert os.path.isabs(result)

        if is_absolute:
            # For absolute paths, the result should be the same as input
            assert result == repo_path
        else:
            # For relative paths, the result should end with the relative part
            # Special handling for paths with .. that get resolved
            if '..' in repo_path:
                # Just verify it's absolute and contains expected path components
                assert os.path.isabs(result)
                # For ../parent/path, we expect it to resolve to a path containing parent/path
                path_parts = repo_path.replace('..', '').strip('/').strip('\\')
                if path_parts:
                    # Normalize both paths for comparison (handles mixed separators)
                    assert os.path.normpath(path_parts) in os.path.normpath(result)
            else:
                # Normalize both paths for comparison (handles mixed separators)
                assert os.path.normpath(result).endswith(os.path.normpath(repo_path))

    def test_feature_store_attribute_access(self, enabled_config: FeatureStoreConfig) -> None:
        """Test direct access to _feature_store attribute."""
        wrapper = FeatureStoreWrapper(enabled_config)

        # Initially should be None
        assert wrapper._feature_store is None

        # After calling get_feature_store, it should be set (but we don't call it to avoid FeatureStore creation)
        # Just verify the attribute exists and is accessible
        assert hasattr(wrapper, '_feature_store')
        assert wrapper._feature_store is None
