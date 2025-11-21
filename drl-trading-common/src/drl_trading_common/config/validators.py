"""Reusable validators for configuration models."""
from pydantic import model_validator

from drl_trading_common.base.base_schema import BaseSchema


class StrictAfterMergeSchema(BaseSchema):
    """Base schema that validates all fields are non-None after instantiation.

    This allows models to have optional fields during YAML loading/merging,
    but ensures all fields are present after the merge is complete.

    Usage:
        class MyConfig(StrictAfterMergeSchema):
            field1: str | None = None
            field2: int | None = None
            field3: bool | None = None

    All fields will be validated as required after instantiation.
    """

    @model_validator(mode='after')
    def validate_all_fields_present(self) -> 'StrictAfterMergeSchema':
        """Ensure all fields are set after YAML merging."""
        missing_fields = []
        for field_name in self.model_fields.keys():
            value = getattr(self, field_name)
            if value is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields in {self.__class__.__name__}: {', '.join(missing_fields)}. "
                f"Ensure these are defined in application.yaml or application-{{stage}}.yaml"
            )

        return self
