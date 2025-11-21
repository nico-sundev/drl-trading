from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BaseParameterSetConfig:
    """Base parameter set configuration as a frozen dataclass."""
    type: str  # discriminator
    enabled: bool

    # Properties assigned during mapping
    hash_id: Optional[str] = field(default=None)
    string_representation: Optional[str] = field(default=None)
