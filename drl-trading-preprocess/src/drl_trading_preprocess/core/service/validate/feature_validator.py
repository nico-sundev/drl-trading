import logging
from typing import Dict, List

from injector import inject

from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_core.core.service.feature_parameter_set_parser import FeatureParameterSetParser

logger = logging.getLogger(__name__)


@inject
class FeatureValidator:
    """Service for validating feature definitions."""

    def __init__(self, feature_manager: FeatureManager,
        feature_parameter_set_parser: FeatureParameterSetParser) -> None:
        """Initialize with feature manager for validation."""
        self.feature_manager = feature_manager
        self.feature_parameter_set_parser = feature_parameter_set_parser

    def validate_definitions(
        self, feature_definitions: List[FeatureDefinition]
    ) -> Dict[str, bool]:
        """Validate multiple feature definitions for support.

        Args:
            feature_definitions: List of feature definitions to validate

        Returns:
            Dictionary mapping feature names to validation status
        """
        logger.debug("Parsing feature parameter sets before validation")
        self.feature_parameter_set_parser.parse_feature_definitions(feature_definitions)

        logger.debug(f"Validating {len(feature_definitions)} feature definitions")
        # Delegate to FeatureManager's validation method
        validation_results = self.feature_manager.validate_feature_definitions(
            feature_definitions
        )

        # Log validation summary
        supported_count = sum(validation_results.values())
        logger.debug(
            f"Feature validation completed: {supported_count}/{len(feature_definitions)} features supported"
        )

        if supported_count < len(feature_definitions):
            unsupported = [
                name for name, supported in validation_results.items() if not supported
            ]
            logger.warning(f"Unsupported features detected: {unsupported}")

        return validation_results
