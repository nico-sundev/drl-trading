import hashlib
import json

from pydantic import BaseModel, ConfigDict, computed_field
from pydantic.alias_generators import to_camel


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
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
