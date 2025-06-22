# drl_trading_common/config/feature_config_repo.py
import json
import logging

import psycopg2.extras
from abc import ABC, abstractmethod
from injector import inject

from drl_trading_common.db.database_connection_interface import DatabaseConnectionInterface
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class FeatureConfigRepoInterface(ABC):
    @abstractmethod
    def get_config(self, version: str) -> FeatureConfigVersionInfo:
        pass

    @abstractmethod
    def is_config_existing(self, version: str) -> bool:
        """
        Check if a feature config with the given version exists.

        Args:
            version (str): The version of the feature config to check.

        Returns:
            bool: True if the config exists, False otherwise.
        """
        pass

    @abstractmethod
    def save_config(self, config: FeatureConfigVersionInfo) -> str:
        pass

@inject
class FeatureConfigPostgresRepo(FeatureConfigRepoInterface):
    """
    PostgreSQL repository for storing and retrieving feature configuration versions.

    This repository manages feature configuration versioning in the database,
    enabling reproducible feature engineering across training and inference.
    """

    def __init__(self, connection_service: DatabaseConnectionInterface):
        """
        Initialize the repository with database connection service.

        Args:
            connection_service: Database connection interface for connection management
        """
        self.connection_service = connection_service
        self.logger = logging.getLogger(__name__)

    def get_config(self, version: str) -> FeatureConfigVersionInfo:
        """
        Retrieve a feature configuration by version.

        Args:
            version: The version identifier (either semver or hash)

        Returns:
            FeatureConfigVersionInfo: The feature configuration for the specified version

        Raises:
            ValueError: If the configuration version is not found
            DatabaseConnectionError: If database operation fails
        """
        try:
            with self.connection_service.get_connection() as connection:
                with connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    # Query by either semver or hash
                    query = """
                        SELECT semver, hash, created_at, feature_definitions, description
                        FROM feature_configs
                        WHERE semver = %s OR hash = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """

                    cursor.execute(query, (version, version))
                    row = cursor.fetchone()

                    if not row:
                        raise ValueError(f"Feature configuration version '{version}' not found")

                    return FeatureConfigVersionInfo(
                        semver=row['semver'],
                        hash=row['hash'],
                        created_at=row['created_at'],
                        feature_definitions=row['feature_definitions'],
                        description=row['description']
                    )

        except ValueError:
            # Re-raise value errors (version not found)
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve config version '{version}': {str(e)}")
            raise

    def save_config(self, config: FeatureConfigVersionInfo) -> str:
        """
        Save a feature configuration version to the database.

        Args:
            config: The feature configuration to save

        Returns:
            str: The version identifier (hash) of the saved configuration

        Raises:
            ValueError: If configuration data is invalid
            DatabaseConnectionError: If database operation fails
        """
        if not config.semver or not config.hash:
            raise ValueError("Configuration must have both semver and hash")

        try:
            with self.connection_service.get_transaction() as cursor:
                # Use UPSERT to handle conflicts gracefully
                upsert_query = """
                    INSERT INTO feature_configs (
                        semver, hash, created_at, feature_definitions, description
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (hash)
                    DO UPDATE SET
                        semver = EXCLUDED.semver,
                        created_at = EXCLUDED.created_at,
                        feature_definitions = EXCLUDED.feature_definitions,
                        description = EXCLUDED.description
                """

                cursor.execute(upsert_query, (
                    config.semver,
                    config.hash,
                    config.created_at,
                    json.dumps(config.feature_definitions),
                    config.description
                ))

                self.logger.info(f"Successfully saved config version {config.semver} (hash: {config.hash})")
                return config.hash

        except Exception as e:
            self.logger.error(f"Failed to save config version {config.semver}: {str(e)}")
            raise

    def is_config_existing(self, version: str) -> bool:
        """
        Check if a feature config with the given version exists.

        Args:
            version: The version of the feature config to check (semver or hash)

        Returns:
            bool: True if the config exists, False otherwise

        Raises:
            DatabaseConnectionError: If database operation fails
        """
        try:
            with self.connection_service.get_connection() as connection:
                with connection.cursor() as cursor:
                    query = """
                        SELECT 1
                        FROM feature_configs
                        WHERE semver = %s OR hash = %s
                        LIMIT 1
                    """

                    cursor.execute(query, (version, version))
                    return cursor.fetchone() is not None

        except Exception as e:
            self.logger.error(f"Failed to check config existence for version '{version}': {str(e)}")
            raise
