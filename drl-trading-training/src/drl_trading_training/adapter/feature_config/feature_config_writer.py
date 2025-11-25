"""
Write adapter for feature configuration data.

This module provides a write adapter for storing feature configuration
data using SQLAlchemy ORM. This adapter is specific to the training service
since it's the only service that writes feature configurations.
"""

import logging
from injector import inject

from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory, SessionFactoryError
from drl_trading_adapter.adapter.database.mapper.feature_config_mapper import FeatureConfigMapper
from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo


class FeatureConfigWriter:
    """
    Write adapter for feature configuration data.

    Provides write operations for feature configurations using SQLAlchemy ORM.
    This adapter is specific to the training service since it's the only service
    that writes feature configurations.
    """

    @inject
    def __init__(self, session_factory: SQLAlchemySessionFactory):
        """
        Initialize the writer with database session factory.

        Args:
            session_factory: SQLAlchemy session factory for database connections
        """
        self.session_factory = session_factory
        self.logger = logging.getLogger(__name__)

    def save_config(self, config: FeatureConfigVersionInfo) -> str:
        """
        Save a feature configuration version to the database.

        Uses UPSERT semantics to handle conflicts gracefully - if a configuration
        with the same hash already exists, it will be updated with the new data.

        Args:
            config: The feature configuration to save

        Returns:
            str: The hash identifier of the saved configuration

        Raises:
            ValueError: If configuration data is invalid
            SessionFactoryError: If database operation fails
        """
        if not config.semver or not config.hash:
            raise ValueError("Configuration must have both semver and hash")

        try:
            with self.session_factory.get_session() as session:
                # Convert domain model to entity
                entity = FeatureConfigMapper.to_entity(config)

                # Use merge for UPSERT semantics
                # If an entity with the same primary key (hash) exists, it will be updated
                # If not, a new entity will be inserted
                merged_entity = session.merge(entity)
                session.commit()

                self.logger.info(
                    f"Successfully saved config version {config.semver} (hash: {config.hash})"
                )
                return merged_entity.hash

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.error(f"Failed to save config version {config.semver}: {str(e)}")
            raise SessionFactoryError(f"Failed to save config version {config.semver}: {str(e)}") from e
