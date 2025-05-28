from typing import List, Optional

from .base_schema import BaseSchema


class ContextFeatureConfig(BaseSchema):
    """
    Configuration class for context features used in the trading environment.

    This class defines the primary and derived context features that are required
    for the trading environment but are not part of the feature set used for machine learning.
    """

    # Primary context columns that must exist in the raw data
    primary_context_columns: List[str]

    # Derived context columns that are computed from primary columns
    derived_context_columns: Optional[List[str]] = None

    # Additional primary context columns that might be used if available
    optional_context_columns: Optional[List[str]] = None

    # Time Column
    time_column: str

    def get_all_primary_columns(self) -> List[str]:
        """
        Returns a list of all primary column names used in the context configuration.

        The returned list includes:
        - The time column (`self.time_column`)
        - All primary context columns (`self.primary_context_columns`)
        - Any optional context columns (`self.optional_context_columns`), if present

        Returns:
            List[str]: A list containing the names of all primary columns.
        """
        return (
            [self.time_column]
            + self.primary_context_columns
            + (self.optional_context_columns if self.optional_context_columns else [])
        )

    def get_all_context_columns(self) -> List[str]:
        """
        Returns a list of all context columns.

        This includes both primary and derived context columns.

        Returns:
            List[str]: List of all context columns
        """
        return (
            self.get_all_primary_columns() + self.derived_context_columns
            if self.derived_context_columns
            else []
        )
