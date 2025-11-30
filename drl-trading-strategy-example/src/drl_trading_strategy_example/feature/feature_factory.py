import logging
from typing import Optional

from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.core.model.base_feature import BaseFeature
from drl_trading_common.interface.feature.feature_factory_interface import (
    IFeatureFactory,
)
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier
from drl_trading_strategy_example.feature.registry.feature_class_registry_interface import (
    IFeatureClassRegistry,
)
from drl_trading_strategy_example.feature.registry.feature_config_registry_interface import (
    IFeatureConfigRegistry,
)
from injector import inject

logger = logging.getLogger(__name__)


@inject
class FeatureFactory(IFeatureFactory):
    """
    Concrete implementation of FeatureFactoryInterface.

    This factory creates feature instances using a registry to obtain class types.
    It implements the actual Factory pattern by creating instances rather than just
    managing class types.
    """

    def __init__(
        self,
        registry: IFeatureClassRegistry,
        config_registry: IFeatureConfigRegistry,
        indicators_service: ITechnicalIndicatorFacade,
    ) -> None:
        """
        Initialize the factory with feature class and config registries.

        Args:
            registry: The feature class registry to use for obtaining class types
            config_registry: The feature config registry to use for obtaining config class types
        """
        self._registry = registry
        self._config_registry = config_registry
        self._technical_indicator_facade = indicators_service

    def create_feature(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ) -> Optional[BaseFeature]:
        """
        Create a feature instance for the given feature name and parameters.

        Args:
            feature_name: The name of the feature to create
            dataset_id: The dataset identifier for the feature
            config: The configuration for the feature (optional)
            postfix: Optional postfix for the feature name

        Returns:
            The created feature instance if successful, None otherwise
        """
        feature_class = self._registry.get_feature_class(feature_name)
        if feature_class is None:
            logger.error(
                f"Feature class for '{feature_name}' not found in registry. Cannot create instance."
            )
            return None

        try:
            # Create the feature instance with the provided parameters
            feature_instance = feature_class(
                dataset_id=dataset_id,
                indicator_service=self._technical_indicator_facade,
                config=config,
                postfix=postfix,
            )
            logger.debug(f"Created feature instance for '{feature_name}'")
            return feature_instance
        except Exception as e:
            logger.error(
                f"Failed to create feature instance for '{feature_name}': {e}",
                exc_info=True,
            )
            return None

    def create_config_instance(
        self, feature_name: str, config_data: dict
    ) -> Optional[BaseParameterSetConfig]:
        """
        Create a configuration instance for the given feature name and data.

        Args:
            feature_name: The name of the feature to create config for
            config_data: The configuration data to initialize the config with

        Returns:
            A configuration instance or None if no config class is found

        Raises:
            ValueError: If the provided config_data is invalid for the config class
        """
        config_class = self._config_registry.get_config_class(feature_name)
        if not config_class:
            logger.debug(f"No config class found for feature '{feature_name}'")
            return None

        try:
            # Create a copy of config_data to avoid modifying the original
            instance_data = config_data.copy()
            # Add type field for discriminated unions if not present
            if "type" not in instance_data:
                instance_data["type"] = feature_name.lower()

            return config_class(**instance_data)
        except Exception as e:
            error_msg = (
                f"Failed to create config instance for '{feature_name}': {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def is_feature_supported(self, feature_name: str) -> bool:
        """
        Check if a feature is fully supported (both class and config available).

        This method orchestrates validation by checking both the feature class
        registry and config registry to ensure complete feature support.

        Args:
            feature_name: The name of the feature to check

        Returns:
            True if the feature can be created successfully, False otherwise
        """
        try:
            # Check if feature class is available
            has_feature_class = self._registry.has_feature_class(feature_name)
            if not has_feature_class:
                logger.debug(f"Feature class not found for '{feature_name}'")
                return False

            # Check if config class is available (if required)
            # Note: Some features might not require configuration
            has_feature_config = self._config_registry.has_feature_config(feature_name)

            # For now, we consider a feature supported if it has a class
            # Config is optional - features can work without configuration
            logger.debug(f"Feature '{feature_name}' validation: class={has_feature_class}, config={has_feature_config}")

            return has_feature_class

        except Exception as e:
            logger.warning(f"Failed to validate feature '{feature_name}': {e}")
            return False
