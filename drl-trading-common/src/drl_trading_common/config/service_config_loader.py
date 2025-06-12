"""Configuration loader with environment detection for microservice deployment."""
import os
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.config_adapter import ConfigAdapter

# Generic type variable for configuration
T = TypeVar('T', bound=BaseApplicationConfig)

logger = logging.getLogger(__name__)


class ServiceConfigLoader:
    """
    Service configuration loader with environment detection.

    This class provides methods to load service-specific configuration files
    with intelligent environment detection and fallback paths.

    Usage:
        # Load a service-specific configuration
        config = ServiceConfigLoader.load_config(
            InferenceConfig,
            service="inference"
        )

        # With explicit path
        config = ServiceConfigLoader.load_config(
            TrainingConfig,
            config_path="path/to/config.yaml"
        )
    """

    DEFAULT_CONFIG_LOCATIONS = [
        "config",               # ./config/ folder in current directory
        "../config",            # ../config/ folder in parent directory
        "src/{service}/config", # Service-specific config directory
        "/app/config"           # Docker container standard location
    ]

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
        env_prefix: Optional[str] = None
    ) -> T:
        """
        Load configuration for a specific service.

        Args:
            config_class: The configuration class to instantiate
            service: Service name (for path calculation)
            config_path: Optional explicit config directory path
            config_file: Optional specific config filename (without extension)
            env_prefix: Optional prefix for environment variables

        Returns:
            Instance of the specified configuration class
        """
        # Check environment variable for config path override
        env_config_path = os.environ.get("SERVICE_CONFIG_PATH")
        if env_config_path and Path(env_config_path).exists():
            logger.info(f"Loading configuration from environment variable: {env_config_path}")
            return ServiceConfigLoader._load_explicit_path(
                config_class,
                env_config_path,
                env_prefix
            )

        # Use explicit path if provided
        if config_path:
            if Path(config_path).is_dir():
                # It's a directory, we need to build the full path
                return ServiceConfigLoader._discover_config_in_directory(
                    config_class,
                    config_path,
                    service,
                    config_file,
                    env_prefix
                )
            elif Path(config_path).is_file():
                # It's a file path, load directly
                return ServiceConfigLoader._load_explicit_path(
                    config_class,
                    config_path,
                    env_prefix
                )

        # Try to discover config in default locations
        service_name = service or config_class.__name__.lower().replace("config", "")

        # Try each default location
        for location in ServiceConfigLoader.DEFAULT_CONFIG_LOCATIONS:
            # Insert service name if placeholder exists
            search_path = location.format(service=service_name)

            if Path(search_path).is_dir():
                try:
                    return ServiceConfigLoader._discover_config_in_directory(
                        config_class,
                        search_path,
                        service_name,
                        config_file,
                        env_prefix
                    )
                except FileNotFoundError:
                    # Try next location
                    continue

        # If we get here, no config was found
        raise FileNotFoundError(
            f"No configuration file found for service {service_name} "
            f"in any of the default locations: {ServiceConfigLoader.DEFAULT_CONFIG_LOCATIONS}"
        )

    @staticmethod
    def _load_explicit_path(
        config_class: Type[T],
        path: str,
        env_prefix: Optional[str] = None
    ) -> T:
        """Load configuration from an explicit file path."""
        env_prefix = env_prefix or config_class.__name__.upper()
        return ConfigAdapter.load_with_env_override(
            config_class,
            path,
            env_prefix=env_prefix
        )

    @staticmethod
    def _discover_config_in_directory(
        config_class: Type[T],
        directory: str,
        service_name: str,
        config_name: Optional[str] = None,
        env_prefix: Optional[str] = None
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

        For each pattern, it tries extensions in order: .yaml, .yml, .json
        """
        env_name = ServiceConfigLoader.get_env_name()
        base_names = []

        if config_name:
            base_names.append(config_name)

        base_names.extend([
            service_name,
            config_class.__name__.lower().replace("config", "")
        ])

        # Add generic "config" as fallback
        base_names.append("config")

        extensions = [".yaml", ".yml", ".json"]

        # Check environment-specific files first
        for base_name in base_names:
            for ext in extensions:
                # Try env-specific file first
                env_file = os.path.join(directory, f"{base_name}.{env_name}{ext}")
                if Path(env_file).exists():
                    logger.info(f"Found environment-specific config: {env_file}")
                    return ServiceConfigLoader._load_explicit_path(
                        config_class,
                        env_file,
                        env_prefix
                    )

                # Then try generic file
                generic_file = os.path.join(directory, f"{base_name}{ext}")
                if Path(generic_file).exists():
                    logger.info(f"Found generic config: {generic_file}")
                    return ServiceConfigLoader._load_explicit_path(
                        config_class,
                        generic_file,
                        env_prefix
                    )

        # No configurations found in this directory
        raise FileNotFoundError(f"No configuration files found in directory {directory}")
