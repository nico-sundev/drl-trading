from dataclasses import dataclass, asdict
import hashlib
import json
from enum import Enum
from typing import Any


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


@dataclass(frozen=True)
class BaseParameterSetConfig:
    """Base parameter set configuration as a frozen dataclass."""
    type: str  # discriminator
    enabled: bool

    @property
    def hash_id(self) -> str:
        """Compute and return the hash ID based on content."""
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True, cls=EnumEncoder)
        return hashlib.md5(config_str.encode()).hexdigest()

    def __hash__(self) -> int:
        """Return hash based on hash_id for content-based hashing."""
        return hash(self.hash_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on hash_id for content-based comparison."""
        if not isinstance(other, BaseParameterSetConfig):
            return False
        return self.hash_id == other.hash_id
