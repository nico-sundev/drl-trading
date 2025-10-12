import logging
from typing import Dict, List

from injector import inject

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_core.core.service.feature_manager import FeatureManager

logger = logging.getLogger(__name__)


@inject
class FeatureValidator:
    """Service for validating feature definitions."""

    def __init__(self, feature_manager: FeatureManager) -> None:
        """Initialize with feature manager for validation."""
        self.feature_manager = feature_manager

    def validate_definitions(
        self, feature_definitions: List[FeatureDefinition]
    ) -> Dict[str, bool]:
        """Validate multiple feature definitions for support.

        Args:
            feature_definitions: List of feature definitions to validate

        Returns:
            Dictionary mapping feature names to validation status
        """
        logger.debug(f"Validating {len(feature_definitions)} feature definitions")

        # Delegate to FeatureManager's validation method
        validation_results = self.feature_manager.validate_feature_definitions(
            feature_definitions
        )

        # Log validation summary
        supported_count = sum(validation_results.values())
        logger.info(
            f"Feature validation completed: {supported_count}/{len(feature_definitions)} features supported"
        )

        if supported_count < len(feature_definitions):
            unsupported = [
                name for name, supported in validation_results.items() if not supported
            ]
            logger.warning(f"Unsupported features detected: {unsupported}")

        return validation_results
