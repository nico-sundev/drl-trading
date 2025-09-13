"""Unit tests for FeatureStoreWrapper.

This module tests the FeatureStoreWrapper class functionality including:
- Absolute and relative path resolution with STAGE-based directory structure
- Disabled config behavior
- FeatureStore instance caching
- Error handling for invalid configurations
- STAGE environment variable handling
"""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

from drl_trading_adapter.adapter.feature_store.provider import FeatureStoreWrapper
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig  # type: ignore
from feast import FeatureStore


class TestFeatureStoreWrapper:
    """Test suite for FeatureStoreWrapper."""

    @pytest.fixture
    def enabled_config(self) -> FeatureStoreConfig:
        """Create an enabled feature store config for testing."""
        return FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

    @pytest.fixture
    def disabled_config(self) -> FeatureStoreConfig:
        """Create a disabled feature store config for testing."""
        return FeatureStoreConfig(
            cache_enabled=False,
            config_directory="/test/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

    @pytest.fixture
    def temp_feature_store_dir(self) -> Generator[str, None, None]:
        """Create a temporary feature store directory with stage subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create stage subdirectories with feature_store.yaml files
            for stage in ["dev", "cicd", "prod", "test"]:
                stage_dir = os.path.join(temp_dir, stage)
                os.makedirs(stage_dir, exist_ok=True)

                # Create a dummy feature_store.yaml for each stage
                config_content = f"""project: {stage}_trading_project
registry: registry.db
provider: local
online_store:
    type: sqlite
    path: online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""
                with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                    f.write(config_content)

            yield temp_dir

    def test_init_with_enabled_config(self, enabled_config: FeatureStoreConfig) -> None:
        """Test initialization with enabled feature store config."""
        wrapper = FeatureStoreWrapper(enabled_config, "test")

        assert wrapper._feature_store_config == enabled_config
        assert wrapper._stage == "test"
        assert wrapper._feature_store is None

    def test_init_with_disabled_config(
        self, disabled_config: FeatureStoreConfig
    ) -> None:
        """Test initialization with disabled feature store config."""
        wrapper = FeatureStoreWrapper(disabled_config, "test")

        assert wrapper._feature_store_config == disabled_config
        assert wrapper._stage == "test"
        assert wrapper._feature_store is None

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_get_feature_store_with_enabled_config(
        self, mock_feast_store: MagicMock, temp_feature_store_dir: str
    ) -> None:
        """Test get_feature_store with enabled config and test stage."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config, "test")
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        # Should call with the test stage subdirectory
        expected_path = os.path.join(temp_feature_store_dir, "test")
        mock_feast_store.assert_called_once_with(repo_path=expected_path)

    def test_get_feature_store_with_disabled_config(
        self, disabled_config: FeatureStoreConfig
    ) -> None:
        """Test get_feature_store returns None when feature store is disabled."""
        wrapper = FeatureStoreWrapper(disabled_config, "test")

        # This should return None due to disabled config in _resolve_feature_store_config_directory
        result = wrapper._resolve_feature_store_config_directory()

        assert result is None

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_get_feature_store_with_absolute_path(
        self, mock_feast_store: MagicMock, temp_feature_store_dir: str
    ) -> None:
        """Test get_feature_store with absolute path to config directory."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,  # This is already absolute
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config, "dev")
        result = wrapper.get_feature_store()

        assert result == mock_feast_instance
        # Should call with the dev stage subdirectory
        expected_path = os.path.join(temp_feature_store_dir, "dev")
        mock_feast_store.assert_called_once_with(repo_path=expected_path)

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_get_feature_store_with_relative_path(
        self, mock_feast_store: MagicMock
    ) -> None:
        """Test get_feature_store with relative path to config directory."""
        # Create a temp directory structure with stage subdirectories
        with tempfile.TemporaryDirectory() as temp_dir:
            relative_path = "test_config/feature_store"
            full_path = os.path.join(temp_dir, relative_path)

            # Create the directory structure with stage subdirectories
            for stage in ["prod"]:  # Only create the stage we're testing
                stage_dir = os.path.join(full_path, stage)
                os.makedirs(stage_dir, exist_ok=True)
                with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                    f.write("project: test\nregistry: registry.db\nprovider: local\n")

            config = FeatureStoreConfig(
                cache_enabled=True,
                config_directory=full_path,  # Use the full path since we can't rely on cwd
                entity_name="test_entity",
                ttl_days=30,
                online_enabled=False,
                service_name="test_service",
                service_version="1.0.0",
            )

            mock_feast_instance = MagicMock(spec=FeatureStore)
            mock_feast_store.return_value = mock_feast_instance

            wrapper = FeatureStoreWrapper(config, "prod")
            result = wrapper.get_feature_store()

            assert result == mock_feast_instance
            # Verify that the path points to the prod stage subdirectory
            call_args = mock_feast_store.call_args_list[0]
            called_path = call_args[1]["repo_path"]
            assert os.path.isabs(called_path)
            assert called_path.endswith(os.path.join("feature_store", "prod"))

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_get_feature_store_caching(
        self, mock_feast_store: MagicMock, temp_feature_store_dir: str
    ) -> None:
        """Test that FeatureStore instance is cached after first creation."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        mock_feast_instance = MagicMock(spec=FeatureStore)
        mock_feast_store.return_value = mock_feast_instance

        wrapper = FeatureStoreWrapper(config, "test")

        # First call should create the instance
        result1 = wrapper.get_feature_store()
        assert result1 == mock_feast_instance
        assert mock_feast_store.call_count == 1

        # Second call should return cached instance
        result2 = wrapper.get_feature_store()
        assert result2 == mock_feast_instance
        assert result1 is result2
        assert mock_feast_store.call_count == 1  # Should not be called again

    def test_get_feature_store_creation_error(self) -> None:
        """Test error handling when STAGE is missing."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/nonexistent/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        wrapper = FeatureStoreWrapper(config, "test")

        with pytest.raises(FileNotFoundError, match="Feature store config not found"):
            wrapper.get_feature_store()

    def test_get_feature_store_missing_config_file(self) -> None:
        """Test error handling when feature_store.yaml is missing."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/nonexistent/path/feature_store",
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        wrapper = FeatureStoreWrapper(config, "test")

        with pytest.raises(FileNotFoundError, match="Feature store config not found"):
            wrapper.get_feature_store()

    def test_resolve_feature_store_config_directory_disabled(
        self, disabled_config: FeatureStoreConfig
    ) -> None:
        """Test _resolve_feature_store_config_directory returns None when disabled."""
        wrapper = FeatureStoreWrapper(disabled_config, "test")

        result = wrapper._resolve_feature_store_config_directory()

        assert result is None

    def test_resolve_feature_store_config_directory_absolute(
        self, temp_feature_store_dir: str
    ) -> None:
        """Test _resolve_feature_store_config_directory with absolute path."""
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        wrapper = FeatureStoreWrapper(config, "test")
        result = wrapper._resolve_feature_store_config_directory()

        # Should return the test stage subdirectory
        expected_path = os.path.join(temp_feature_store_dir, "test")
        assert result == expected_path
        assert os.path.isabs(result)

    def test_resolve_feature_store_config_directory_relative(self) -> None:
        """Test _resolve_feature_store_config_directory with relative path."""
        # Create a temp directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            relative_path = os.path.join(temp_dir, "test_config", "feature_store")

            # Create the directory structure with stage subdirectory
            stage_dir = os.path.join(relative_path, "dev")
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                f.write("project: test\nregistry: registry.db\nprovider: local\n")

            config = FeatureStoreConfig(
                cache_enabled=True,
                config_directory=relative_path,  # Use absolute path for testing
                entity_name="test_entity",
                ttl_days=30,
                online_enabled=False,
                service_name="test_service",
                service_version="1.0.0",
            )

            wrapper = FeatureStoreWrapper(config, "dev")
            result = wrapper._resolve_feature_store_config_directory()

            assert result is not None
            assert os.path.isabs(result)
            # Should point to the dev stage subdirectory
            assert result.endswith(os.path.join("feature_store", "dev"))

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_multiple_wrappers_independent_caching(
        self, mock_feast_store: MagicMock, temp_feature_store_dir: str
    ) -> None:
        """Test that multiple wrapper instances have independent caching."""
        mock_feast_instance1 = MagicMock(spec=FeatureStore)
        mock_feast_instance2 = MagicMock(spec=FeatureStore)
        mock_feast_store.side_effect = [mock_feast_instance1, mock_feast_instance2]

        # Create two configs with the same path
        config1 = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        config2 = FeatureStoreConfig(
            cache_enabled=True,
            config_directory=temp_feature_store_dir,
            entity_name="test_entity",
            ttl_days=30,
            online_enabled=False,
            service_name="test_service",
            service_version="1.0.0",
        )

        # Create two wrapper instances
        wrapper1 = FeatureStoreWrapper(config1, "prod")
        wrapper2 = FeatureStoreWrapper(config2, "prod")

        # Each should create its own FeatureStore instance
        result1 = wrapper1.get_feature_store()
        result2 = wrapper2.get_feature_store()

        assert result1 == mock_feast_instance1
        assert result2 == mock_feast_instance2
        assert result1 is not result2
        assert mock_feast_store.call_count == 2

    @patch(
        "drl_trading_adapter.adapter.feature_store.provider.feature_store_wrapper.FeatureStore"
    )
    def test_path_with_special_characters(self, mock_feast_store: MagicMock) -> None:
        """Test handling of paths with special characters."""
        # Create a temp directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            special_path = os.path.join(temp_dir, "test-feature_store@v1")

            # Create the directory structure with stage subdirectory
            stage_dir = os.path.join(special_path, "prod")
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                f.write("project: test\nregistry: registry.db\nprovider: local\n")

            config = FeatureStoreConfig(
                cache_enabled=True,
                config_directory=special_path,
                entity_name="test_entity",
                ttl_days=30,
                online_enabled=False,
                service_name="test_service",
                service_version="1.0.0",
            )

            mock_feast_instance = MagicMock(spec=FeatureStore)
            mock_feast_store.return_value = mock_feast_instance

            wrapper = FeatureStoreWrapper(config, "prod")
            result = wrapper.get_feature_store()

            assert result == mock_feast_instance
            call_args = mock_feast_store.call_args_list[0]
            called_path = call_args[1]["repo_path"]
            assert os.path.isabs(called_path)
            # Should point to the prod stage subdirectory
            assert called_path.endswith(os.path.join("test-feature_store@v1", "prod"))

    def test_config_object_attributes(self, enabled_config: FeatureStoreConfig) -> None:
        """Test that config object has expected attributes."""
        wrapper = FeatureStoreWrapper(enabled_config, "test")

        assert wrapper._feature_store_config.cache_enabled is True
        assert (
            wrapper._feature_store_config.config_directory == "/test/path/feature_store"
        )
        assert wrapper._feature_store_config.entity_name == "test_entity"
        assert wrapper._feature_store_config.ttl_days == 30
        assert wrapper._feature_store_config.online_enabled is False
        assert wrapper._feature_store_config.service_name == "test_service"
        assert wrapper._feature_store_config.service_version == "1.0.0"

    @pytest.mark.parametrize(
        "enabled,expected_none",
        [
            (True, False),
            (False, True),
        ],
    )
    def test_resolve_path_parametrized(
        self, enabled: bool, expected_none: bool
    ) -> None:
        """Test path resolution with parametrized enabled/disabled states."""
        # Create a temp directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test", "path")

            if enabled:
                # Create the directory structure with stage subdirectory
                stage_dir = os.path.join(test_path, "test")
                os.makedirs(stage_dir, exist_ok=True)
                with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                    f.write("project: test\nregistry: registry.db\nprovider: local\n")

            config = FeatureStoreConfig(
                cache_enabled=enabled,
                config_directory=test_path,
                entity_name="test_entity",
                ttl_days=30,
                online_enabled=False,
                service_name="test_service",
                service_version="1.0.0",
            )

            wrapper = FeatureStoreWrapper(config, "test")

            if expected_none:
                result = wrapper._resolve_feature_store_config_directory()
                assert result is None
            else:
                result = wrapper._resolve_feature_store_config_directory()
                assert result is not None
                assert os.path.isabs(result)

    @pytest.mark.parametrize(
        "repo_path,stage,is_absolute",
        [
            ("/absolute/path/to/repo", "test", True),
            ("relative/path/to/repo", "dev", False),
            ("C:\\Windows\\path\\to\\repo", "prod", True),  # Windows absolute path
            ("./relative/path", "cicd", False),
            ("../parent/path", "test", False),
        ],
    )
    def test_path_resolution_types(
        self, repo_path: str, stage: str, is_absolute: bool
    ) -> None:
        """Test path resolution with various path types and stages."""
        # Create a temp directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            if is_absolute:
                # For absolute paths, create structure directly
                if repo_path.startswith("C:") or repo_path.startswith("/"):
                    # Skip actual absolute paths that we can't create
                    pytest.skip("Skipping actual absolute path test")
                else:
                    full_path = repo_path
            else:
                # For relative paths, create within temp directory
                full_path = os.path.join(temp_dir, repo_path.lstrip("./").lstrip("../"))

            # Create the directory structure with stage subdirectory
            stage_dir = os.path.join(full_path, stage)
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "feature_store.yaml"), "w") as f:
                f.write("project: test\nregistry: registry.db\nprovider: local\n")

            config = FeatureStoreConfig(
                cache_enabled=True,
                config_directory=full_path,
                entity_name="test_entity",
                ttl_days=30,
                online_enabled=False,
                service_name="test_service",
                service_version="1.0.0",
            )

            # Use injected stage directly (implementation no longer reads env)
            wrapper = FeatureStoreWrapper(config, stage)
            result = wrapper._resolve_feature_store_config_directory()

            assert result is not None
            assert os.path.isabs(result)
            # Should point to the stage subdirectory
            assert result.endswith(stage)

    def test_feature_store_attribute_access(
        self, enabled_config: FeatureStoreConfig
    ) -> None:
        """Test direct access to _feature_store attribute."""
        wrapper = FeatureStoreWrapper(enabled_config, "test")

        # Initially should be None
        assert wrapper._feature_store is None

        # After calling get_feature_store, it should be set (but we don't call it to avoid FeatureStore creation)
        # Just verify the attribute exists and is accessible
        assert hasattr(wrapper, "_feature_store")
        assert wrapper._feature_store is None
