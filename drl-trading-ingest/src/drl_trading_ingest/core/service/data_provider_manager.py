"""Manager for coordinating access to data providers."""

import logging
from typing import Dict, List, Optional

from drl_trading_ingest.core.port import DataProviderPort

logger = logging.getLogger(__name__)


class DataProviderManager:
    """Manages access to registered data providers.

    This manager provides a clean interface for services to access data providers
    without depending on concrete adapter implementations.
    """

    def __init__(self) -> None:
        """Initialize the manager with an empty provider registry."""
        self._providers: Dict[str, DataProviderPort] = {}

    def register_provider(self, name: str, provider: DataProviderPort) -> None:
        """
        Register a data provider.

        Args:
            name: Unique name for the provider (e.g., "csv", "binance")
            provider: Provider instance implementing DataProviderPort
        """
        if name in self._providers:
            logger.warning(f"Overriding existing provider: {name}")

        self._providers[name] = provider
        logger.debug(f"Registered provider: {name}")

    def get_provider(self, name: str) -> Optional[DataProviderPort]:
        """
        Get a specific data provider by name.

        Args:
            name: Name of the provider

        Returns:
            Provider instance or None if not found
        """
        provider = self._providers.get(name)

        if not provider:
            logger.warning(
                f"Provider '{name}' not found. "
                f"Available: {', '.join(self._providers.keys())}"
            )

        return provider

    def get_all_providers(self) -> Dict[str, DataProviderPort]:
        """
        Get all registered providers.

        Returns:
            Dictionary mapping provider names to provider instances
        """
        return self._providers.copy()

    def get_available_provider_names(self) -> List[str]:
        """
        Get list of available provider names.

        Returns:
            List of registered provider names
        """
        return list(self._providers.keys())

    def teardown_all(self) -> None:
        """Teardown all registered providers."""
        logger.info(f"Tearing down {len(self._providers)} providers")

        for name, provider in self._providers.items():
            try:
                provider.teardown()
                logger.debug(f"Teardown completed for provider: {name}")
            except Exception as e:
                logger.error(f"Error tearing down provider '{name}': {e}")

        self._providers.clear()
