import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


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


class BaseParameterSetConfig(BaseModel):
    type: str  # discriminator
    enabled: bool

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

    def hash_id(self) -> str:
        """Generate a unique hash for this parameter set configuration

        Returns:
            str: MD5 hash of the configuration as a hexadecimal string
        """
        config_dict = self.model_dump(exclude={"hash_id"})
        config_str = json.dumps(config_dict, sort_keys=True, cls=EnumEncoder)
        return hashlib.md5(config_str.encode()).hexdigest()

    def to_string(self, max_length: int = 50) -> str:
        """Generate a human-readable string representation of this parameter set.

        The string is formed by concatenating key parameter values with underscores.
        Complex nested structures are flattened, and the string is truncated if it
        exceeds max_length.

        Args:
            max_length: Maximum length of the generated string. Defaults to 50.

        Returns:
            str: A human-readable string representation of the parameter set
        """
        result_parts = []

        # Get all fields except internal ones
        config_dict = self.model_dump(exclude={"hash_id"})

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
