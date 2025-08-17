"""Lean service configuration loader for application.yaml with stage overrides."""
import os
import re
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar, Dict, Any, Union

from drl_trading_common.logging.bootstrap_logging import get_bootstrap_logger

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.config_adapter import ConfigAdapter

# Generic type variable for configuration
T = TypeVar('T', bound=BaseApplicationConfig)

logger = logging.getLogger(__name__)


class ServiceConfigLoader:
    """
    Lean configuration loader for service application configs.

    Loads: .env file + application.yaml + application-{STAGE}.yaml + secret substitution
    """

    # Pattern for secret substitution: ${SECRET_NAME:default_value}
    SECRET_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    @staticmethod
    def load_config(config_class: Type[T], service_name: Optional[str] = None) -> T:
        """
        Load service configuration with stage overrides and secret substitution.

        Args:
            config_class: The configuration class to instantiate

        Returns:
            Instance of the specified configuration class
        """
        # Load .env file if present (for local development)
        ServiceConfigLoader._load_dotenv()

        svc_name = service_name or config_class.__name__.lower()
        bootstrap_logger = get_bootstrap_logger(svc_name)

        # Get config directory from environment
        config_dir = os.environ.get("CONFIG_DIR")
        if not config_dir:
            bootstrap_logger.error(
                "CONFIG_DIR environment variable not set; cannot load configuration"
            )
            raise ValueError(
                "CONFIG_DIR environment variable must be set. "
                "Example: CONFIG_DIR=/path/to/config or CONFIG_DIR=./config"
            )

        if not Path(config_dir).exists():
            bootstrap_logger.error("Configuration directory not found: %s", config_dir)
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

        # Get stage from environment
        stage = os.environ.get("STAGE", "local")
        bootstrap_logger.info(
            "Loading config (service=%s, stage=%s, dir=%s)", svc_name, stage, config_dir
        )

        # Find base application config file (YAML preference)
        base_file = ServiceConfigLoader._find_config_file(config_dir, "application")
        if not base_file:
            bootstrap_logger.error(
                "Base configuration file application.yaml not found in %s", config_dir
            )
            raise FileNotFoundError(f"Base configuration file not found: application.yaml in {config_dir}")
        bootstrap_logger.info("Base configuration file: %s", base_file)

        # Find stage override file (optional)
        stage_file = ServiceConfigLoader._find_config_file(config_dir, f"application-{stage}")
        if stage != "local" and not stage_file:
            bootstrap_logger.warning(
                "Stage override file application-%s.yaml not found (continuing with base)",
                stage,
            )

        # Load base configuration
        config = ServiceConfigLoader._load_single_file(config_class, base_file)

        # Apply stage override if exists
        if stage_file:
            bootstrap_logger.info("Applying stage override: %s", stage_file)
            # Load stage override as raw YAML data (not as config object)
            import yaml
            with open(stage_file, 'r') as f:
                stage_data = yaml.safe_load(f)

            # Apply secret substitution to stage data
            if stage_data:
                stage_data = ServiceConfigLoader._substitute_secrets(stage_data)

                # Merge stage data into base config
                config_dict = config.model_dump()
                ServiceConfigLoader._deep_update(config_dict, stage_data)
                before_keys = set(config.model_dump().keys())
                config = config_class.model_validate(config_dict)
                after_keys = set(config.model_dump().keys())
                changed = after_keys.union(before_keys)
                bootstrap_logger.info(
                    "Stage override applied (file=%s, keys=%d)", stage_file, len(changed)
                )

        # Provide warning if logging section missing for observability
        if not hasattr(config, "logging"):
            bootstrap_logger.warning(
                "No logging section present in config object (defaults will be used)"
            )
        return config

    @staticmethod
    def _deep_update(base_dict: dict, update_dict: dict) -> None:
        """
        Deep update base_dict with update_dict, modifying base_dict in place.

        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates to apply
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                ServiceConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    @staticmethod
    def _find_config_file(config_dir: str, filename: str) -> Optional[str]:
        """Find YAML config file."""
        file_path = os.path.join(config_dir, f"{filename}.yaml")
        if Path(file_path).exists():
            return file_path
        return None

    @staticmethod
    def _load_single_file(config_class: Type[T], file_path: str) -> T:
        """Load configuration from a single file with secret substitution."""
        # Load using ConfigAdapter with environment variable overrides
        env_prefix = config_class.__name__.upper()
        config = ConfigAdapter.load_with_env_override(config_class, file_path, env_prefix=env_prefix)

        # Apply secret substitution
        config_dict = config.model_dump()
        config_dict = ServiceConfigLoader._substitute_secrets(config_dict)
        return config_class.model_validate(config_dict)

    @staticmethod
    def _substitute_secrets(data: Union[Dict[str, Any], Any]) -> Union[Dict[str, Any], Any]:
        """
        Replace secret placeholders with environment variables.

        Supports syntax: ${SECRET_NAME:default_value}
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
                        raise ValueError(f"Required secret '{secret_name}' not found in environment and no default provided")

                return ServiceConfigLoader.SECRET_PATTERN.sub(replace_secret, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(v) for v in value]
            else:
                return value

        return substitute_value(data)

    @staticmethod
    def _load_dotenv() -> None:
        """
        Load .env file from current working directory if it exists.

        Uses python-dotenv if available, falls back to manual parsing if not.
        This enables local development without requiring explicit environment setup.
        """
        env_file = Path(".env")
        if not env_file.exists():
            return

        try:
            # Try to use python-dotenv if available
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)  # Don't override existing env vars
            logger.debug(f"Loaded .env file using dotenv: {env_file}")
        except ImportError:
            # Fall back to manual parsing if python-dotenv not available
            logger.debug("python-dotenv not available, using manual .env parsing")
