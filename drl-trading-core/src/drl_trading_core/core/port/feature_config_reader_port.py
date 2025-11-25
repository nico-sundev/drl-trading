"""
Feature configuration reader port for shared read-only access.

This port defines the contract for reading feature configurations across services.
It provides read-only access to feature configuration data without write operations,
supporting configuration retrieval for all services that need feature config access.
"""

from abc import ABC, abstractmethod
from typing import Optional

from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo


class FeatureConfigReaderPort(ABC):
    """
    Interface for read-only feature configuration access.

    This port enables multiple services to read feature configurations from the shared
    feature configuration storage while maintaining clear separation from write operations
    that remain exclusive to the ingest service.
    """

    @abstractmethod
    def get_config(self, version: str) -> FeatureConfigVersionInfo:
        """
        Retrieve a feature configuration by version identifier.

        Args:
            version: The version identifier (either semver or hash)

        Returns:
            FeatureConfigVersionInfo: The feature configuration for the specified version

        Raises:
            ValueError: If the configuration version is not found
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def is_config_existing(self, version: str) -> bool:
        """
        Check if a feature configuration with the given version exists.

        Args:
            version: The version identifier (either semver or hash)

        Returns:
            bool: True if the configuration exists, False otherwise

        Raises:
            DatabaseConnectionError: If database access fails
        """
        pass

    @abstractmethod
    def get_latest_config_by_semver_prefix(self, semver_prefix: str) -> Optional[FeatureConfigVersionInfo]:
        """
        Get the latest configuration matching a semantic version prefix.

        This method is useful for getting the latest patch version of a specific
        major.minor version (e.g., "1.2" would return the latest "1.2.x" version).

        Args:
            semver_prefix: The semantic version prefix to match (e.g., "1.2", "1.2.3")

        Returns:
            FeatureConfigVersionInfo or None: The latest matching configuration,
            or None if no match is found

        Raises:
            DatabaseConnectionError: If database access fails
        """
        pass
