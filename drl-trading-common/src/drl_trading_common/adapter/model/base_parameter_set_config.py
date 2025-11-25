"""
Base parameter set configuration for adapter layer.

This is the adapter representation of parameter set configurations,
used for external communication and serialization. It contains the
essential fields needed for cross-service communication.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseParameterSetConfig(BaseModel):
    """Base parameter set configuration for adapter layer."""
    type: str  # discriminator
    enabled: bool

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
