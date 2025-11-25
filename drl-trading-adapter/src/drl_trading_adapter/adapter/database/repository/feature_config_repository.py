"""
SQLAlchemy-based feature configuration repository implementation.

This module provides Entity Framework-style repository implementation using
SQLAlchemy ORM for feature configuration data access.
"""

import logging
from typing import Optional
from sqlalchemy import or_
from injector import inject

from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory, SessionFactoryError
from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity
from drl_trading_adapter.adapter.database.mapper.feature_config_mapper import FeatureConfigMapper
from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo
from drl_trading_core.core.port.feature_config_reader_port import FeatureConfigReaderPort


@inject
class FeatureConfigRepository(FeatureConfigReaderPort):
    """
    SQLAlchemy-based feature configuration repository.

    Provides Entity Framework-style data access using SQLAlchemy ORM.
    Implements the read-only port for shared access across services.
    """

    def __init__(self, session_factory: SQLAlchemySessionFactory) -> None:
        """
        Initialize repository with SQLAlchemy session factory.

        Args:
            session_factory: SQLAlchemy session factory for database access
        """
        self.session_factory = session_factory
        self.logger = logging.getLogger(__name__)

    def get_config(self, version: str) -> FeatureConfigVersionInfo:
        """
        Retrieve a feature configuration by version identifier.

        Args:
            version: The version identifier (either semver or hash)

        Returns:
            FeatureConfigVersionInfo: The feature configuration for the specified version

        Raises:
            ValueError: If the configuration version is not found
            SessionFactoryError: If database operation fails
        """
        entity = None
        try:
            with self.session_factory.get_read_only_session() as session:
                # Query by either semver or hash, order by created_at DESC to get latest
                entity = session.query(FeatureConfigEntity).filter(
                    or_(
                        FeatureConfigEntity.semver == version,
                        FeatureConfigEntity.hash == version
                    )
                ).order_by(FeatureConfigEntity.created_at.desc()).first()

                # Check if entity was found and convert within session context
                if not entity:
                    raise ValueError(f"Feature configuration version '{version}' not found")

                # Convert to domain model within session context to access all attributes
                return FeatureConfigMapper.to_domain_model(entity)

        except SessionFactoryError as e:
            # Check if this is actually a ValueError wrapped by SessionFactory
            if "not found" in str(e):
                raise ValueError(f"Feature configuration version '{version}' not found")
            # Otherwise it's a real database error
            self.logger.error(f"Failed to retrieve config version '{version}': {str(e)}")
            raise
        except ValueError:
            # Re-raise ValueError as-is (from our entity check above)
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve config version '{version}': {str(e)}")
            raise SessionFactoryError(f"Failed to retrieve config version '{version}': {str(e)}") from e

    def is_config_existing(self, version: str) -> bool:
        """
        Check if a feature configuration with the given version exists.

        Args:
            version: The version identifier (either semver or hash)

        Returns:
            bool: True if the configuration exists, False otherwise

        Raises:
            SessionFactoryError: If database operation fails
        """
        try:
            with self.session_factory.get_read_only_session() as session:
                # Check if any configuration exists with the given version
                exists = session.query(FeatureConfigEntity).filter(
                    or_(
                        FeatureConfigEntity.semver == version,
                        FeatureConfigEntity.hash == version
                    )
                ).first() is not None

                return exists

        except Exception as e:
            self.logger.error(f"Failed to check config existence for version '{version}': {str(e)}")
            raise SessionFactoryError(f"Failed to check config existence for version '{version}': {str(e)}") from e

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
            SessionFactoryError: If database operation fails
        """
        try:
            with self.session_factory.get_read_only_session() as session:
                # Find configurations with semver starting with the prefix
                entity = session.query(FeatureConfigEntity).filter(
                    FeatureConfigEntity.semver.like(f"{semver_prefix}%")
                ).order_by(FeatureConfigEntity.created_at.desc()).first()

                if not entity:
                    return None

                return FeatureConfigMapper.to_domain_model(entity)

        except Exception as e:
            self.logger.error(f"Failed to get latest config for prefix '{semver_prefix}': {str(e)}")
            raise SessionFactoryError(f"Failed to get latest config for prefix '{semver_prefix}': {str(e)}") from e
