import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field
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

    @computed_field
    def hash_id(self) -> str:
        """Generate a unique hash for this parameter set configuration"""
        config_dict = self.model_dump(exclude={"hash_id"})
        config_str = json.dumps(config_dict, sort_keys=True, cls=EnumEncoder)
        return hashlib.md5(config_str.encode()).hexdigest()
