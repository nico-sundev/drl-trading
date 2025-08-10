"""Config adapters for multiple file formats and sources (JSON, YAML, ENV)."""
import os
from typing import Optional, Type, TypeVar

import yaml

from drl_trading_common.base.base_application_config import BaseApplicationConfig

# Type variable bound to BaseApplicationConfig for generic methods
T = TypeVar('T', bound=BaseApplicationConfig)


class ConfigAdapter:
    """
    Configuration loading and adaptation between multiple sources.

    This class provides methods to load configuration from different sources
    (JSON, YAML, ENV) and convert between formats as needed.
    """

    @staticmethod
    def load_from_json(config_class: Type[T], path: str) -> T:
        """
        Load configuration from a JSON file.

        Args:
            config_class: The configuration class to instantiate
            path: Path to the JSON configuration file

        Returns:
            Instance of the specified configuration class
        """
        with open(path) as f:
            return config_class.model_validate_json(f.read())

    @staticmethod
    def load_from_yaml(config_class: Type[T], path: str) -> T:
        """
        Load configuration from a YAML file.

        Args:
            config_class: The configuration class to instantiate
            path: Path to the YAML configuration file

        Returns:
            Instance of the specified configuration class
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return config_class.model_validate(data)

    @staticmethod
    def load_with_env_override(
        config_class: Type[T],
        path: str,
        env_prefix: Optional[str] = None
    ) -> T:
        """
        Load configuration with environment variable overrides.

        This method loads configuration from a file and then overrides values
        with environment variables if they exist. Environment variables should
        follow the format {env_prefix}__{field_name} where field_name matches
        the field in the configuration class.

        Args:
            config_class: The configuration class to instantiate
            path: Path to the configuration file (can be JSON or YAML)
            env_prefix: Optional prefix for environment variables

        Returns:
            Instance of the specified configuration class with overrides applied
        """
        # Detect file format from extension
        if path.lower().endswith('.json'):
            config = ConfigAdapter.load_from_json(config_class, path)
        elif path.lower().endswith(('.yaml', '.yml')):
            config = ConfigAdapter.load_from_yaml(config_class, path)
        else:
            raise ValueError(f"Unsupported config file format: {path}")

        # Apply environment variable overrides if env_prefix is provided
        if env_prefix:
            config_dict = config.model_dump()

            # Get all environment variables
            env_vars = dict(os.environ.items())

            # Look for environment variables with the specified prefix
            for env_var, value in env_vars.items():
                if env_var.startswith(f"{env_prefix}__"):
                    # Extract the field path from the environment variable name
                    field_path = env_var[len(f"{env_prefix}__"):].lower()

                    # Navigate the nested dict structure and set the value
                    parts = field_path.split('__')

                    # Find the right part of the config to update
                    target = config_dict
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]

                    # Set the value, converting to appropriate type based on config class
                    target[parts[-1]] = value

            # Revalidate with overrides applied
            config = config_class.model_validate(config_dict)

        return config
