"""Service configuration loader with YAML preference and secret substitution support."""
import os
import re
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar, Dict, Any, Union

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.config_adapter import ConfigAdapter

# Generic type variable for configuration
T = TypeVar('T', bound=BaseApplicationConfig)

logger = logging.getLogger(__name__)


class EnhancedServiceConfigLoader:
    """
    Service configuration loader with YAML preference and advanced features.

    This class provides methods to load service-specific configuration files
    with intelligent environment detection and fallback paths, featuring:
    - YAML-first configuration loading
    - Secret substitution using ${VAR_NAME:default} syntax
    - Enhanced environment variable override support
    - Improved error messages and logging

    Usage:
        # Basic usage
        config = EnhancedServiceConfigLoader.load_config(
            InferenceConfig,
            service="inference"
        )

        # With secret substitution and environment overrides
        config = EnhancedServiceConfigLoader.load_config(
            InferenceConfig,
            service="inference",
            secret_substitution=True,
            env_override=True
        )
    """

    DEFAULT_CONFIG_LOCATIONS = [
        "config",               # ./config/ folder in current directory
        "../config",            # ../config/ folder in parent directory
        "src/{service}/config", # Service-specific config directory
        "/app/config"           # Docker container standard location
    ]

    # YAML-first extension preference
    PREFERRED_EXTENSIONS = [".yaml", ".yml", ".json"]

    # Pattern for secret substitution: ${SECRET_NAME:default_value}
    SECRET_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    @staticmethod
    def get_env_name() -> str:
        """Get the current deployment environment name."""
        return os.environ.get("DEPLOYMENT_MODE", "development")

    @staticmethod
    def load_config(
        config_class: Type[T],
        service: Optional[str] = None,
        config_path: Optional[str] = None,
        config_file: Optional[str] = None,
        env_prefix: Optional[str] = None,
        secret_substitution: bool = True,
        env_override: bool = True
    ) -> T:
        """
        Load configuration for a specific service with enhanced features.

        Args:
            config_class: The configuration class to instantiate
            service: Service name (for path calculation)
            config_path: Optional explicit config directory path
            config_file: Optional specific config filename (without extension)
            env_prefix: Optional prefix for environment variables
            secret_substitution: Enable secret substitution (default: True)
            env_override: Enable environment variable overrides (default: True)

        Returns:
            Instance of the specified configuration class
        """
        # Check environment variable for config path override
        env_config_path = os.environ.get("SERVICE_CONFIG_PATH")
        if env_config_path and Path(env_config_path).exists():
            logger.info(f"Loading configuration from environment variable: {env_config_path}")
            return EnhancedServiceConfigLoader._load_explicit_path(
                config_class,
                env_config_path,
                env_prefix,
                secret_substitution,
                env_override
            )

        # Use explicit path if provided
        if config_path:
            if Path(config_path).is_dir():
                # It's a directory, we need to build the full path
                return EnhancedServiceConfigLoader._discover_config_in_directory(
                    config_class,
                    config_path,
                    service,
                    config_file,
                    env_prefix,
                    secret_substitution,
                    env_override
                )
            elif Path(config_path).is_file():
                # It's a file path, load directly
                return EnhancedServiceConfigLoader._load_explicit_path(
                    config_class,
                    config_path,
                    env_prefix,
                    secret_substitution,
                    env_override
                )

        # Try to discover config in default locations
        service_name = service or config_class.__name__.lower().replace("config", "")

        # Try each default location
        for location in EnhancedServiceConfigLoader.DEFAULT_CONFIG_LOCATIONS:
            # Insert service name if placeholder exists
            search_path = location.format(service=service_name)

            if Path(search_path).is_dir():
                try:
                    return EnhancedServiceConfigLoader._discover_config_in_directory(
                        config_class,
                        search_path,
                        service_name,
                        config_file,
                        env_prefix,
                        secret_substitution,
                        env_override
                    )
                except FileNotFoundError:
                    # Try next location
                    continue

        # If we get here, no config was found
        raise FileNotFoundError(
            f"No configuration file found for service {service_name} "
            f"in any of the default locations: {EnhancedServiceConfigLoader.DEFAULT_CONFIG_LOCATIONS}"
        )

    @staticmethod
    def _load_explicit_path(
        config_class: Type[T],
        path: str,
        env_prefix: Optional[str] = None,
        secret_substitution: bool = True,
        env_override: bool = True
    ) -> T:
        """Load configuration from an explicit file path with enhanced features."""
        env_prefix = env_prefix or config_class.__name__.upper()

        # Load base configuration using ConfigAdapter
        config = ConfigAdapter.load_with_env_override(
            config_class,
            path,
            env_prefix=env_prefix if env_override else None
        )

        # Apply secret substitution if enabled
        if secret_substitution:
            config_dict = config.model_dump()
            config_dict = EnhancedServiceConfigLoader._substitute_secrets(config_dict)
            config = config_class.model_validate(config_dict)

        return config

    @staticmethod
    def _discover_config_in_directory(
        config_class: Type[T],
        directory: str,
        service_name: str,
        config_name: Optional[str] = None,
        env_prefix: Optional[str] = None,
        secret_substitution: bool = True,
        env_override: bool = True
    ) -> T:
        """
        Discover configuration file in a directory with environment-specific variants.

        This method checks for files in the following order:
        1. {config_name}.{env}.{ext}
        2. {service_name}.{env}.{ext}
        3. {config_class_name}.{env}.{ext}
        4. {config_name}.{ext}
        5. {service_name}.{ext}
        6. {config_class_name}.{ext}
        7. config.{env}.{ext}
        8. config.{ext}

        For each pattern, it tries extensions in order: .yaml, .yml, .json (YAML preference)
        """
        env_name = EnhancedServiceConfigLoader.get_env_name()
        base_names = []

        if config_name:
            base_names.append(config_name)

        base_names.extend([
            service_name,
            config_class.__name__.lower().replace("config", "")
        ])

        # Add generic "config" as fallback
        base_names.append("config")

        # YAML-first extension preference
        extensions = EnhancedServiceConfigLoader.PREFERRED_EXTENSIONS

        # Check environment-specific files first
        for base_name in base_names:
            for ext in extensions:
                # Try env-specific file first
                env_file = os.path.join(directory, f"{base_name}.{env_name}{ext}")
                if Path(env_file).exists():
                    logger.info(f"Found environment-specific config: {env_file}")
                    return EnhancedServiceConfigLoader._load_explicit_path(
                        config_class,
                        env_file,
                        env_prefix,
                        secret_substitution,
                        env_override
                    )

                # Then try generic file
                generic_file = os.path.join(directory, f"{base_name}{ext}")
                if Path(generic_file).exists():
                    logger.info(f"Found generic config: {generic_file}")
                    return EnhancedServiceConfigLoader._load_explicit_path(
                        config_class,
                        generic_file,
                        env_prefix,
                        secret_substitution,
                        env_override
                    )

        # No configurations found in this directory
        raise FileNotFoundError(f"No configuration files found in directory {directory}")

    @staticmethod
    def _substitute_secrets(data: Union[Dict[str, Any], Any]) -> Union[Dict[str, Any], Any]:
        """
        Replace secret placeholders with environment variables.

        Supports syntax: ${SECRET_NAME:default_value}
        - SECRET_NAME: Environment variable name
        - default_value: Optional default value if environment variable is not set

        Examples:
            ${DB_PASSWORD}                    # Required secret, no default
            ${DB_PASSWORD:default_pass}       # Secret with default value
            ${API_KEY:}                       # Secret with empty string default
        """
        def substitute_value(value: Any) -> Any:
            if isinstance(value, str):
                def replace_secret(match: re.Match) -> str:
                    secret_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""

                    env_value = os.environ.get(secret_name)
                    if env_value is not None:
                        logger.debug(f"Substituted secret: {secret_name}")
                        return env_value
                    elif default_value or match.group(2) is not None:
                        logger.debug(f"Using default value for secret: {secret_name}")
                        return default_value
                    else:
                        logger.error(f"Secret {secret_name} not found in environment and no default provided")
                        raise ValueError(f"Required secret '{secret_name}' not found in environment and no default provided")

                return EnhancedServiceConfigLoader.SECRET_PATTERN.sub(replace_secret, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(v) for v in value]
            else:
                return value

        return substitute_value(data)

    @staticmethod
    def validate_config_file(config_path: str) -> bool:
        """
        Validate that a configuration file exists and is readable.

        Args:
            config_path: Path to configuration file

        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.error(f"Configuration file does not exist: {config_path}")
                return False

            if not path.is_file():
                logger.error(f"Configuration path is not a file: {config_path}")
                return False

            if path.suffix.lower() not in EnhancedServiceConfigLoader.PREFERRED_EXTENSIONS:
                logger.warning(f"Configuration file has unsupported extension: {config_path}")
                return False

            # Try to read the file
            with open(path, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.error(f"Configuration file is empty: {config_path}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating configuration file {config_path}: {e}")
            return False

    @staticmethod
    def list_available_configs(directory: str) -> Dict[str, list]:
        """
        List all available configuration files in a directory.

        Args:
            directory: Directory to search for configuration files

        Returns:
            Dictionary mapping config types to list of available files
        """
        configs = {"yaml": [], "json": [], "other": []}

        try:
            path = Path(directory)
            if not path.exists() or not path.is_dir():
                return configs

            for file_path in path.iterdir():
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in [".yaml", ".yml"]:
                        configs["yaml"].append(str(file_path))
                    elif suffix == ".json":
                        configs["json"].append(str(file_path))
                    else:
                        configs["other"].append(str(file_path))

        except Exception as e:
            logger.error(f"Error listing configuration files in {directory}: {e}")

        return configs
