from dataclasses import dataclass

from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_common.model.timeframe import Timeframe


@dataclass
class FeatureServiceMetadata:

    feature_service_role: FeatureRoleEnum
    symbol: str
    feature_version_info: FeatureConfigVersionInfo
    timeframe: Timeframe

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
        if not isinstance(self.feature_service_role, FeatureRoleEnum):
            raise ValueError(
                f"Feature role must be a FeatureRoleEnum, got {type(self.feature_service_role)}"
            )

        # Feature validation
        if not isinstance(self.feature_version_info, FeatureConfigVersionInfo):
            raise ValueError(
                f"Feature must be a FeatureConfigVersionInfo, got {type(self.feature_version_info)}"
            )

        # Validate feature definitions exist
        if not self.feature_version_info.feature_definitions:
            raise ValueError("Feature definitions must not be empty")

        # Timeframe validation
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError(
                f"Timeframe must be a Timeframe, got {type(self.timeframe)}"
            )

    def get_sanitized_symbol(self) -> str:
        """Get sanitized symbol string."""
        return self.symbol.strip() if self.symbol else ""

    @classmethod
    def create(
        cls,
        symbol: str,
        timeframe: Timeframe,
        feature_role: FeatureRoleEnum,
        feature_config_version: FeatureConfigVersionInfo,
    ) -> "FeatureServiceMetadata":
        """
        Factory method to create and validate a FeatureServiceRequestContainer.

        Args:
            symbol: The trading symbol for the feature view
            timeframe: The timeframe for the feature view
            feature_role: The role of the feature in the view
            feature_config_version: The feature configuration version info

        Returns:
            FeatureServiceRequestContainer: Validated request object

        Raises:
            ValueError: If any parameter is invalid
        """
        request = cls(
            symbol=symbol,
            feature_service_role=feature_role,
            feature_version_info=feature_config_version,
            timeframe=timeframe,
        )
        request.validate()
        return request
