"""
Mapper for BaseParameterSetConfig between adapter and core layers.
"""

import hashlib
import json
from enum import Enum
from typing import Any

from drl_trading_common.adapter.model.base_parameter_set_config import (
    BaseParameterSetConfig as AdapterBaseParameterSetConfig,
)
from drl_trading_common.core.model.base_parameter_set_config import (
    BaseParameterSetConfig as CoreBaseParameterSetConfig,
)


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum types by using their values."""

    def default(self, obj: Any) -> Any:
        """Override default encoding behavior for custom types.

        Args:
            obj: The object to encode

        Returns:
            A JSON serializable representation of the object
        """
        if isinstance(obj, Enum):
            return obj.value
        # Let the base class handle anything we don't explicitly handle
        return super().default(obj)


class BaseParameterSetConfigMapper:
    """Mapper for BaseParameterSetConfig between adapter and core layers."""

    @staticmethod
    def dto_to_domain(
        dto: AdapterBaseParameterSetConfig,
    ) -> CoreBaseParameterSetConfig:
        """Convert adapter BaseParameterSetConfig (DTO) to core BaseParameterSetConfig (domain).

        Args:
            dto: BaseParameterSetConfig from adapter layer (DTO)

        Returns:
            Corresponding core BaseParameterSetConfig domain model with computed fields
        """
        # Generate hash_id
        config_dict = dto.model_dump(exclude={"hash_id"})
        config_str = json.dumps(config_dict, sort_keys=True, cls=EnumEncoder)
        hash_id = hashlib.md5(config_str.encode()).hexdigest()

        # Generate string representation
        string_representation = BaseParameterSetConfigMapper._generate_string_representation(
            dto
        )

        return CoreBaseParameterSetConfig(
            type=dto.type,
            enabled=dto.enabled,
            hash_id=hash_id,
            string_representation=string_representation,
        )

    @staticmethod
    def domain_to_dto(
        domain: CoreBaseParameterSetConfig,
    ) -> AdapterBaseParameterSetConfig:
        """Convert core BaseParameterSetConfig (domain) to adapter BaseParameterSetConfig (DTO).

        Args:
            domain: BaseParameterSetConfig from core layer (domain)

        Returns:
            Corresponding adapter BaseParameterSetConfig DTO without computed fields
        """
        return AdapterBaseParameterSetConfig(
            type=domain.type,
            enabled=domain.enabled,
        )

    @staticmethod
    def _generate_string_representation(
        config: AdapterBaseParameterSetConfig, max_length: int = 50
    ) -> str:
        """Generate a human-readable string representation of this parameter set.

        The string is formed by concatenating key parameter values with underscores.
        Complex nested structures are flattened, and the string is truncated if it
        exceeds max_length.

        Args:
            config: The parameter set configuration
            max_length: Maximum length of the generated string. Defaults to 50.

        Returns:
            str: A human-readable string representation of the parameter set
        """
        result_parts = []

        # Get all fields except internal ones
        config_dict = config.model_dump(exclude={"hash_id"})

        # Filter out common fields that don't add distinguishing information
        if "type" in config_dict:
            del config_dict["type"]
        if "enabled" in config_dict:
            del config_dict["enabled"]

        # Process and add each parameter value
        for key, value in config_dict.items():
            # Skip empty or None values
            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                continue

            # Format the value based on its type
            if isinstance(value, bool):
                # For booleans, only add the parameter name if True
                if value:
                    result_parts.append(key)
            elif isinstance(value, (int, float)):
                # For numbers, add key=value
                result_parts.append(f"{value}")
            elif isinstance(value, str):
                result_parts.append(value)
            elif isinstance(value, (list, tuple)):
                # For lists, add the length
                result_parts.append(f"{key}{len(value)}")
            elif isinstance(value, dict):
                # For dicts, add a count
                result_parts.append(f"{key}{len(value)}")
            else:
                # Default case - just add the key
                result_parts.append(key)

        # Join all parts with underscores
        result = "_".join(result_parts)

        # Truncate if too long, preserving start and end
        if len(result) > max_length and max_length >= 10:
            half = (max_length - 3) // 2
            result = f"{result[:half]}...{result[-half:]}"

        return result
