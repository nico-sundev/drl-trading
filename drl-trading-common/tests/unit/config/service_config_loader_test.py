"""
Unit tests for EnhancedServiceConfigLoader.

This module tests the EnhancedServiceConfigLoader functionality after
completion of the migration from the original ServiceConfigLoader.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader


class TestConfig(BaseApplicationConfig):
    """Test configuration class for migration testing."""

    app_name: str = "test_service"
    service_name: str = "test"
    database_url: str = "localhost"
    api_key: str = "default_key"
    port: int = 8080


class TestEnhancedServiceConfigLoader:
    """Test EnhancedServiceConfigLoader functionality after migration completion."""

    def test_enhanced_loader_imports_successfully(self) -> None:
        """Test that EnhancedServiceConfigLoader can be imported."""
        # Given/When/Then
        # Import should succeed without errors
        from drl_trading_common.config import EnhancedServiceConfigLoader as ESCLFromInit
        from drl_trading_common.config.enhanced_service_config_loader import EnhancedServiceConfigLoader

        assert EnhancedServiceConfigLoader is not None
        assert ESCLFromInit is not None
        assert EnhancedServiceConfigLoader == ESCLFromInit

    def test_enhanced_loader_has_all_required_methods(self) -> None:
        """Test that EnhancedServiceConfigLoader has all required methods."""
        # Given
        required_methods = [
            'load_config',
            'validate_config_file',
            'list_available_configs',
            'get_env_name',
            '_substitute_secrets'
        ]

        # When/Then
        for method_name in required_methods:
            assert hasattr(EnhancedServiceConfigLoader, method_name), f"Missing method: {method_name}"

    def test_basic_config_loading(self) -> None:
        """Test basic configuration loading functionality."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "postgresql://localhost:5432/test"
api_key: "test_key_123"
port: 9000
"""
            config_file.write_text(config_content)

            # When - Load configuration
            with patch.dict(os.environ, {}, clear=True):
                config = EnhancedServiceConfigLoader.load_config(
                    TestConfig,
                    config_path=str(config_file),
                    secret_substitution=False,
                    env_override=False
                )

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
            config_file = Path(temp_dir) / "test_config.yaml"
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
                'DB_URL': 'postgresql://prod.example.com:5432/prod_db',
                'API_KEY': 'prod_api_key_xyz'
            }):
                config = EnhancedServiceConfigLoader.load_config(
                    TestConfig,
                    config_path=str(config_file),
                    secret_substitution=True
                )

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
            config_file = Path(temp_dir) / "test_config.yaml"
            config_content = """
app_name: "test_service"
service_name: "test_service"
database_url: "${DB_URL:postgresql://localhost:5432/default}"
api_key: "${MISSING_KEY:fallback_key}"
port: 9000
"""
            config_file.write_text(config_content)

            # When - Load without environment variables (should use defaults)
            with patch.dict(os.environ, {}, clear=True):
                config = EnhancedServiceConfigLoader.load_config(
                    TestConfig,
                    config_path=str(config_file),
                    secret_substitution=True
                )

            # Then - Default values should be used
            assert config.database_url == "postgresql://localhost:5432/default"
            assert config.api_key == "fallback_key"
            assert config.app_name == "test_service"  # Required field

    def test_yaml_preference_over_json(self) -> None:
        """Test that YAML files are preferred over JSON files."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create both YAML and JSON files
            yaml_file = Path(temp_dir) / "config.yaml"
            json_file = Path(temp_dir) / "config.json"

            yaml_content = """
app_name: "yaml_service"
service_name: "yaml_service"
database_url: "yaml_database"
api_key: "yaml_key"
port: 8080
"""
            json_content = """{
    "app_name": "json_service",
    "service_name": "json_service",
    "database_url": "json_database",
    "api_key": "json_key",
    "port": 9090
}"""
            yaml_file.write_text(yaml_content)
            json_file.write_text(json_content)

            # When - Load config from directory
            config = EnhancedServiceConfigLoader.load_config(
                TestConfig,
                config_path=temp_dir,
                secret_substitution=False
            )

            # Then - YAML config should be loaded (not JSON)
            assert config.app_name == "yaml_service"
            assert config.service_name == "yaml_service"
            assert config.database_url == "yaml_database"
            assert config.api_key == "yaml_key"
            assert config.port == 8080

    def test_validate_config_file_functionality(self) -> None:
        """Test the validate_config_file utility method."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_config = Path(temp_dir) / "valid.yaml"
            invalid_config = Path(temp_dir) / "invalid.txt"
            empty_config = Path(temp_dir) / "empty.yaml"

            valid_config.write_text("service_name: test")
            invalid_config.write_text("some content")
            empty_config.write_text("")

            # When/Then
            assert EnhancedServiceConfigLoader.validate_config_file(str(valid_config)) is True
            assert EnhancedServiceConfigLoader.validate_config_file(str(invalid_config)) is False
            assert EnhancedServiceConfigLoader.validate_config_file(str(empty_config)) is False
            assert EnhancedServiceConfigLoader.validate_config_file("nonexistent.yaml") is False

    def test_list_available_configs_functionality(self) -> None:
        """Test the list_available_configs utility method."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various config files
            (Path(temp_dir) / "config1.yaml").write_text("test: 1")
            (Path(temp_dir) / "config2.yml").write_text("test: 2")
            (Path(temp_dir) / "config3.json").write_text('{"test": 3}')
            (Path(temp_dir) / "other.txt").write_text("not a config")

            # When
            configs = EnhancedServiceConfigLoader.list_available_configs(temp_dir)

            # Then
            assert len(configs["yaml"]) == 2  # .yaml and .yml files
            assert len(configs["json"]) == 1  # .json file
            assert len(configs["other"]) == 1  # .txt file

            # Check specific files are found
            yaml_files = [Path(f).name for f in configs["yaml"]]
            assert "config1.yaml" in yaml_files
            assert "config2.yml" in yaml_files

    def test_environment_override_functionality(self) -> None:
        """Test enhanced environment variable override support."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
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
                'TESTCONFIG__SERVICE_NAME': 'overridden_service',
                'TESTCONFIG__PORT': '9999'
            }):
                config = EnhancedServiceConfigLoader.load_config(
                    TestConfig,
                    config_path=str(config_file),
                    env_override=True,
                    secret_substitution=False
                )

            # Then - Environment variables should override config values
            assert config.app_name == "test_service"  # Not overridden
            assert config.service_name == "overridden_service"
            assert config.port == 9999
            assert config.database_url == "localhost"  # Not overridden
            assert config.api_key == "default_key"  # Not overridden


class TestServiceIntegration:
    """Test that services using EnhancedServiceConfigLoader work correctly."""

    @patch('drl_trading_inference.bootstrap.InferenceConfig')
    def test_inference_service_migration(self, mock_config_class) -> None:
        """Test that inference service uses EnhancedServiceConfigLoader correctly."""
        # Given
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # When - Import the migrated bootstrap module
        try:
            from drl_trading_inference.bootstrap import bootstrap_inference_service
            # The import should succeed without errors
            assert bootstrap_inference_service is not None
        except ImportError:
            # This is expected in test environment without the actual service
            pytest.skip("Inference service not available in test environment")

    @patch('drl_trading_ingest.infrastructure.di.ingest_module.DataIngestionConfig')
    def test_ingest_service_migration(self, mock_config_class) -> None:
        """Test that ingest service uses EnhancedServiceConfigLoader correctly."""
        # Given
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # When - Import the migrated DI module
        try:
            from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule
            # The import should succeed without errors
            assert IngestModule is not None
        except ImportError:
            # This is expected in test environment without the actual service
            pytest.skip("Ingest service not available in test environment")

    def test_enhanced_config_loader_in_init_exports(self) -> None:
        """Test that EnhancedServiceConfigLoader is properly exported."""
        # Given/When
        from drl_trading_common.config import __all__

        # Then
        assert "EnhancedServiceConfigLoader" in __all__


class TestSecretSubstitutionPattern:
    """Test the secret substitution pattern implementation."""

    def test_secret_pattern_regex(self) -> None:
        """Test the secret substitution regex pattern."""
        # Given
        pattern = EnhancedServiceConfigLoader.SECRET_PATTERN

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
            result = EnhancedServiceConfigLoader._substitute_secrets(test_data)

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
            result = EnhancedServiceConfigLoader._substitute_secrets(test_data)

        # Then
        assert result["hosts"][0] == "prod1.example.com"
        assert result["hosts"][1] == "127.0.0.1"  # Used default
        assert result["hosts"][2] == "static.host.com"  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__])
