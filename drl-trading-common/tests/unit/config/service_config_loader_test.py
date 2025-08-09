"""
Unit tests for EnhancedServiceConfigLoader.

This module tests the EnhancedServiceConfigLoader functionality after
completion of the migration from the original ServiceConfigLoader.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.service_config_loader import ServiceConfigLoader


class TestConfig(BaseApplicationConfig):
    """Test configuration class for migration testing."""

    app_name: str = "test_service"
    service_name: str = "test_service"
    database_url: str = "localhost"
    api_key: str = "default_key"
    port: int = 8080


class TestEnhancedServiceConfigLoader:
    """Test EnhancedServiceConfigLoader functionality after migration completion."""

    def test_enhanced_loader_imports_successfully(self) -> None:
        """Test that EnhancedServiceConfigLoader can be imported."""
        # Given/When/Then
        # Import should succeed without errors
        from drl_trading_common.config import ServiceConfigLoader as ESCLFromInit
        from drl_trading_common.config.service_config_loader import ServiceConfigLoader

        assert ServiceConfigLoader is not None
        assert ESCLFromInit is not None
        assert ServiceConfigLoader == ESCLFromInit

    def test_enhanced_loader_has_all_required_methods(self) -> None:
        """Test that EnhancedServiceConfigLoader has all required methods."""
        # Given
        required_methods = [
            'load_config',
            '_substitute_secrets'
        ]

        # When/Then
        for method_name in required_methods:
            assert hasattr(ServiceConfigLoader, method_name), f"Missing method: {method_name}"

    def test_basic_config_loading(self) -> None:
        """Test basic configuration loading functionality."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "application.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "postgresql://localhost:5432/test"
api_key: "test_key_123"
port: 9000
"""
            config_file.write_text(config_content)

            # When - Load configuration
            with patch.dict(os.environ, {'CONFIG_DIR': temp_dir, 'STAGE': 'local'}, clear=True):
                config = ServiceConfigLoader.load_config(TestConfig)

            # Then - Configuration should be loaded correctly
            assert config.app_name == "test_service"
            assert config.service_name == "test_service"
            assert config.database_url == "postgresql://localhost:5432/test"
            assert config.api_key == "test_key_123"
            assert config.port == 9000

    def test_secret_substitution_functionality(self) -> None:
        """Test that secret substitution works correctly."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "application.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "${DB_URL:postgresql://localhost:5432/default}"
api_key: "${API_KEY}"
port: 9000
"""
            config_file.write_text(config_content)

            # When - Load with environment variables set
            with patch.dict(os.environ, {
                'CONFIG_DIR': temp_dir,
                'STAGE': 'local',
                'DB_URL': 'postgresql://prod.example.com:5432/prod_db',
                'API_KEY': 'prod_api_key_xyz'
            }):
                config = ServiceConfigLoader.load_config(TestConfig)

            # Then - Environment variables should be substituted
            assert config.database_url == "postgresql://prod.example.com:5432/prod_db"
            assert config.api_key == "prod_api_key_xyz"
            assert config.service_name == "test_service"  # No substitution
            assert config.app_name == "test_service"  # Required field
            assert config.port == 9000

    def test_secret_substitution_with_defaults(self) -> None:
        """Test secret substitution with default values."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "application.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "${DB_URL:postgresql://localhost:5432/default}"
api_key: "${MISSING_KEY:fallback_key}"
port: 9000
"""
            config_file.write_text(config_content)

            # When - Load without environment variables (should use defaults)
            with patch.dict(os.environ, {'CONFIG_DIR': temp_dir, 'STAGE': 'local'}, clear=True):
                config = ServiceConfigLoader.load_config(TestConfig)

            # Then - Default values should be used
            assert config.database_url == "postgresql://localhost:5432/default"
            assert config.api_key == "fallback_key"
            assert config.app_name == "test_service"  # Required field

    def test_stage_override_functionality(self) -> None:
        """Test that stage-specific overrides work correctly."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create base config
            base_config = Path(temp_dir) / "application.yaml"
            base_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "localhost:5432"
api_key: "default_key"
port: 8080
"""
            base_config.write_text(base_content)

            # Create stage override
            stage_config = Path(temp_dir) / "application-prod.yaml"
            stage_content = """
database_url: "prod.example.com:5432"
api_key: "prod_key"
port: 9090
"""
            stage_config.write_text(stage_content)

            # When - Load with stage override
            with patch.dict(os.environ, {'CONFIG_DIR': temp_dir, 'STAGE': 'prod'}):
                config = ServiceConfigLoader.load_config(TestConfig)

            # Then - Stage overrides should be applied
            assert config.app_name == "test_service"  # From base
            assert config.service_name == "test_service"  # From base
            assert config.database_url == "prod.example.com:5432"  # Overridden
            assert config.api_key == "prod_key"  # Overridden
            assert config.port == 9090  # Overridden

    def test_environment_override_functionality(self) -> None:
        """Test enhanced environment variable override support."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "application.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "localhost"
api_key: "default_key"
port: 8080
"""
            config_file.write_text(config_content)

            # When - Load with environment overrides
            with patch.dict(os.environ, {
                'CONFIG_DIR': temp_dir,
                'STAGE': 'local',
                'TESTCONFIG__SERVICE_NAME': 'overridden_service',
                'TESTCONFIG__PORT': '9999'
            }):
                config = ServiceConfigLoader.load_config(TestConfig)

            # Then - Environment variables should override config values
            assert config.app_name == "test_service"  # Not overridden
            assert config.service_name == "overridden_service"
            assert config.port == 9999
            assert config.database_url == "localhost"  # Not overridden
            assert config.api_key == "default_key"  # Not overridden


class TestServiceIntegration:
    """Test that EnhancedServiceConfigLoader integrates properly with the common package."""

    def test_enhanced_config_loader_in_init_exports(self) -> None:
        """Test that ServiceConfigLoader is properly exported."""
        # Given/When
        from drl_trading_common.config import __all__

        # Then
        assert "ServiceConfigLoader" in __all__


class TestSecretSubstitutionPattern:
    """Test the secret substitution pattern implementation."""

    def test_secret_pattern_regex(self) -> None:
        """Test the secret substitution regex pattern."""
        # Given
        pattern = ServiceConfigLoader.SECRET_PATTERN

        test_cases = [
            ("${VAR_NAME}", ("VAR_NAME", None)),
            ("${VAR_NAME:default}", ("VAR_NAME", "default")),
            ("${API_KEY:}", ("API_KEY", "")),
            ("${DB_PASSWORD:very_secure_123}", ("DB_PASSWORD", "very_secure_123")),
        ]

        # When/Then
        for test_string, expected in test_cases:
            match = pattern.search(test_string)
            assert match is not None, f"Pattern should match: {test_string}"
            assert match.group(1) == expected[0], f"Variable name mismatch for: {test_string}"
            assert match.group(2) == expected[1], f"Default value mismatch for: {test_string}"

    def test_substitute_secrets_method(self) -> None:
        """Test the _substitute_secrets method directly."""
        # Given
        test_data = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "password": "${DB_PASSWORD}",
                "port": 5432
            },
            "api_key": "${API_KEY:default_key}",
            "service_name": "test_service"
        }

        # When
        with patch.dict(os.environ, {
            'DB_HOST': 'prod.database.com',
            'DB_PASSWORD': 'super_secret',
            # API_KEY not set, should use default
        }):
            result = ServiceConfigLoader._substitute_secrets(test_data)

        # Then
        assert result["database"]["host"] == "prod.database.com"
        assert result["database"]["password"] == "super_secret"
        assert result["database"]["port"] == 5432  # Unchanged
        assert result["api_key"] == "default_key"  # Used default
        assert result["service_name"] == "test_service"  # Unchanged

    def test_substitute_secrets_with_lists(self) -> None:
        """Test secret substitution with list values."""
        # Given
        test_data = {
            "hosts": [
                "${HOST1:localhost}",
                "${HOST2:127.0.0.1}",
                "static.host.com"
            ]
        }

        # When
        with patch.dict(os.environ, {'HOST1': 'prod1.example.com'}):
            result = ServiceConfigLoader._substitute_secrets(test_data)

        # Then
        assert result["hosts"][0] == "prod1.example.com"
        assert result["hosts"][1] == "127.0.0.1"  # Used default
        assert result["hosts"][2] == "static.host.com"  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__])
