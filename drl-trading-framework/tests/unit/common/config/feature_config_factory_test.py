"""
Unit tests for feature_config_factory module.

Tests the functionality of the FeatureConfigFactory class, including discovery,
registration, and creation of feature configuration instances.
"""

from unittest import mock

import pytest
from drl_trading_common.config.base_parameter_set_config import BaseParameterSetConfig

from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)


class TestFeatureConfigFactory:
    """Tests for the FeatureConfigFactory class."""

    def test_get_config_class_not_found(self):
        """Test getting a config class that doesn't exist."""
        # Given
        factory = FeatureConfigFactory()

        # When
        result = factory.get_config_class("nonexistent")

        # Then
        assert result is None

    def test_register_and_get_config_class(self):
        """Test registering and retrieving a config class."""
        # Given
        factory = FeatureConfigFactory()

        # Create a mock config class
        class MockConfig(BaseParameterSetConfig):
            length: int = 14

        # When
        factory.register_config_class("mock", MockConfig)
        retrieved_class = factory.get_config_class("mock")

        # Then
        assert retrieved_class is MockConfig

    def test_register_validates_base_class(self):
        """Test that registering a non-BaseParameterSetConfig class raises TypeError."""
        # Given
        factory = FeatureConfigFactory()

        # Create a class that doesn't extend BaseParameterSetConfig
        class InvalidClass:
            pass

        # When/Then
        with pytest.raises(TypeError, match="must extend BaseParameterSetConfig"):
            factory.register_config_class("invalid", InvalidClass)

    def test_register_case_insensitive(self):
        """Test that feature name registration is case-insensitive."""
        # Given
        factory = FeatureConfigFactory()

        # Create a mock config class
        class MockConfig(BaseParameterSetConfig):
            length: int = 14

        # When
        factory.register_config_class("MOCK", MockConfig)

        # Then
        assert factory.get_config_class("mock") is MockConfig
        assert factory.get_config_class("Mock") is MockConfig
        assert factory.get_config_class("MOCK") is MockConfig

    def test_register_override_warning(self, caplog):
        """Test that a warning is logged when overriding an existing config class."""
        # Given
        factory = FeatureConfigFactory()

        class FirstConfig(BaseParameterSetConfig):
            length: int = 14

        class SecondConfig(BaseParameterSetConfig):
            period: int = 7

        factory.register_config_class("test", FirstConfig)

        # When
        with caplog.at_level("WARNING"):
            factory.register_config_class("test", SecondConfig)

        # Then
        assert "Overriding existing config class" in caplog.text
        assert factory.get_config_class("test") is SecondConfig

    def test_clear(self):
        """Test clearing all registered config classes."""
        # Given
        factory = FeatureConfigFactory()

        class MockConfig(BaseParameterSetConfig):
            length: int = 14

        factory.register_config_class("mock", MockConfig)
        assert factory.get_config_class("mock") is not None

        # When
        factory.clear()

        # Then
        assert factory.get_config_class("mock") is None

    @mock.patch("importlib.import_module")
    @mock.patch("pkgutil.iter_modules")
    @mock.patch("inspect.getmembers")
    def test_discover_config_classes(
        self,
        mock_getmembers: mock.MagicMock,
        mock_iter_modules: mock.MagicMock,
        mock_import_module: mock.MagicMock,
    ) -> None:
        """Test discovering config classes in a package."""
        # Given
        factory = FeatureConfigFactory()

        # Create mock config classes
        class Feat1Config(BaseParameterSetConfig):
            length: int = 14

        class Feat2Config(BaseParameterSetConfig):
            period: int = 7

        # Mock package path
        mock_package = mock.MagicMock()
        mock_package.__path__ = ["/fake/path"]
        mock_import_module.return_value = mock_package

        # Mock module iteration
        mock_iter_modules.return_value = [
            (None, "module1", False),
            (None, "module2", True),  # is_pkg=True should be skipped
        ]

        # Mock module inspection
        mock_module1 = mock.MagicMock()
        mock_import_module.side_effect = [mock_package, mock_module1]

        # Mock class discovery
        mock_getmembers.return_value = [
            ("NotAConfig", type("NotAConfig", (), {})),
            ("Feat1Config", Feat1Config),
            ("Feat2Config", Feat2Config),
            ("BaseParameterSetConfig", BaseParameterSetConfig),
        ]

        # When
        factory.discover_config_classes("test.package")

        # Then
        mock_import_module.assert_any_call("test.package")
        mock_import_module.assert_any_call("test.package.module1")

        # Verify that import_module was NOT called with "test.package.module2"
        assert not any(
            call_args[0][0] == "test.package.module2"
            for call_args in mock_import_module.call_args_list
        ), "import_module should not have been called with 'test.package.module2'"

        # Check if classes were registered correctly
        assert factory.get_config_class("feat1") is Feat1Config
        assert factory.get_config_class("feat2") is Feat2Config

    def test_create_config_instance(self):
        """Test creating a config instance from data."""
        # Given
        factory = FeatureConfigFactory()

        class TestConfig(BaseParameterSetConfig):
            length: int = 14
            enabled: bool = True

        factory.register_config_class("test", TestConfig)

        # When
        instance = factory.create_config_instance("test", {"length": 21})

        # Then
        assert instance is not None
        assert instance.length == 21
        assert instance.enabled is True  # Default value

    def test_create_config_instance_with_invalid_data(self):
        """Test creating a config instance with invalid data raises ValueError."""
        # Given
        factory = FeatureConfigFactory()

        class TestConfig(BaseParameterSetConfig):
            length: int

        factory.register_config_class("test", TestConfig)

        # When/Then
        with pytest.raises(ValueError):
            factory.create_config_instance("test", {})  # Missing required field

    def test_create_config_instance_nonexistent_feature(self):
        """Test creating a config instance for a nonexistent feature returns None."""
        # Given
        factory = FeatureConfigFactory()

        # When
        instance = factory.create_config_instance("nonexistent", {"field": "value"})

        # Then
        assert instance is None
