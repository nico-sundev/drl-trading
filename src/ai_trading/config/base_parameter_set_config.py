from pydantic.alias_generators import to_camel
from pydantic import BaseModel, ConfigDict

class BaseParameterSetConfig(BaseModel):
    type: str  # discriminator
    enabled: bool
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
