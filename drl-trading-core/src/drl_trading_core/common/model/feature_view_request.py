from dataclasses import dataclass

from drl_trading_common.base import BaseFeature
from drl_trading_common.enum import FeatureRoleEnum


@dataclass
class FeatureViewRequestContainer:
    """
    Container for feature view creation parameters.

    This encapsulates all the information needed to create a Feast feature view,
    improving readability and maintainability by grouping related parameters.

    Attributes:
        symbol: The trading symbol for the feature view
        feature_role: Role/type of features
        feature: The feature to include in the view
    """

    symbol: str
    feature_role: FeatureRoleEnum
    feature: BaseFeature

    def validate(self) -> None:
        """
        Validate all parameters in the request.

        Raises:
            ValueError: If any parameter is invalid or missing required attributes
        """
        # Symbol validation
        if (
            not self.symbol
            or not isinstance(self.symbol, str)
            or not self.symbol.strip()
        ):
            raise ValueError("Symbol must be a non-empty string")

        # Feature role validation
        if not isinstance(self.feature_role, FeatureRoleEnum):
            raise ValueError(
                f"Feature role must be a FeatureRoleEnum, got {type(self.feature_role)}"
            )

        # Feature validation
        if not isinstance(self.feature, BaseFeature):
            raise ValueError(f"Feature must be a BaseFeature, got {type(self.feature)}")

    def get_sanitized_symbol(self) -> str:
        """Get sanitized symbol string."""
        return self.symbol.strip() if self.symbol else ""

    @classmethod
    def create(
        cls, symbol: str, feature_role: FeatureRoleEnum, feature: BaseFeature
    ) -> "FeatureViewRequestContainer":
        """
        Factory method to create and validate a FeatureViewRequest.

        Args:
            symbol: The trading symbol for the feature view
            feature_role: Role/type of features (or None for integration tests)
            feature: The feature to include in the view

        Returns:
            FeatureViewRequest: Validated request object

        Raises:
            ValueError: If any parameter is invalid
        """
        request = cls(symbol=symbol, feature_role=feature_role, feature=feature)
        request.validate()
        return request
