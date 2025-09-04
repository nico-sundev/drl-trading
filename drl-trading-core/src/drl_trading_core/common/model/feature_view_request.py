from dataclasses import dataclass
from typing import Optional

from drl_trading_common.base import BaseFeature
from drl_trading_common.enum import FeatureRoleEnum
from drl_trading_common.model import (
    FeatureConfigVersionInfo,
)


@dataclass
class FeatureViewRequest:
    """
    Container for feature view creation parameters.

    This encapsulates all the information needed to create a Feast feature view,
    improving readability and maintainability by grouping related parameters.

    Attributes:
        symbol: The trading symbol for the feature view
        feature_view_name: Name of the feature view to create
        feature_role: Role/type of features (or None for integration tests)
        feature_version_info: Version and metadata information
    """
    symbol: str
    feature_view_name: str
    feature_role: Optional[FeatureRoleEnum]
    feature_version_info: FeatureConfigVersionInfo
    features: list[BaseFeature]

    def validate(self) -> None:
        """
        Validate all parameters in the request.

        Raises:
            ValueError: If any parameter is invalid or missing required attributes
        """
        # Symbol validation
        if not self.symbol or not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("Symbol must be a non-empty string")

        # Feature view name validation
        if not self.feature_view_name or not isinstance(self.feature_view_name, str) or not self.feature_view_name.strip():
            raise ValueError("Feature view name must be a non-empty string")

        # Feature role validation (allow None for integration test compatibility)
        if self.feature_role is not None and not isinstance(self.feature_role, FeatureRoleEnum):
            raise ValueError(f"Feature role must be a FeatureRoleEnum or None, got {type(self.feature_role)}")

        # Feature version info validation
        if self.feature_version_info is None:
            raise ValueError("Feature version info cannot be None")

        # Features length validation
        if self.features is None or len(self.features) == 0:
            raise ValueError("Features list cannot be None or empty")

        # Validate feature version info has required attributes
        if not hasattr(self.feature_version_info, 'semver') or not self.feature_version_info.semver:
            raise ValueError("Feature version info must have a valid semver")

        if not hasattr(self.feature_version_info, 'hash') or not self.feature_version_info.hash:
            raise ValueError("Feature version info must have a valid hash")

    def get_sanitized_symbol(self) -> str:
        """Get sanitized symbol string."""
        return self.symbol.strip() if self.symbol else ""

    def get_sanitized_feature_view_name(self) -> str:
        """Get sanitized feature view name string."""
        return self.feature_view_name.strip() if self.feature_view_name else ""

    def get_role_description(self) -> str:
        """Get human-readable role description for logging."""
        return self.feature_role.value if self.feature_role else "None"

    @classmethod
    def create(
        cls,
        symbol: str,
        feature_view_name: str,
        feature_role: Optional[FeatureRoleEnum],
        feature_version_info: FeatureConfigVersionInfo,
        features: list[BaseFeature]
    ) -> "FeatureViewRequest":
        """
        Factory method to create and validate a FeatureViewRequest.

        Args:
            symbol: The trading symbol for the feature view
            feature_view_name: Name of the feature view to create
            feature_role: Role/type of features (or None for integration tests)
            feature_version_info: Version and metadata information
            features: List of role features to include in the view

        Returns:
            FeatureViewRequest: Validated request object

        Raises:
            ValueError: If any parameter is invalid
        """
        request = cls(
            symbol=symbol,
            feature_view_name=feature_view_name,
            feature_role=feature_role,
            feature_version_info=feature_version_info,
            features=features
        )
        request.validate()
        return request
