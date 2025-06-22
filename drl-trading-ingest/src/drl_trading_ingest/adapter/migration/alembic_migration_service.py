"""
Alembic-based implementation of the migration service interface.

This module provides a concrete implementation of the MigrationServiceInterface
using Alembic as the underlying migration framework. It handles database schema
evolution for the trading ingestion service.
"""

import logging
import os
from typing import List, Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from injector import inject
from sqlalchemy import create_engine

from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationError,
    MigrationServiceInterface,
)
from drl_trading_ingest.infrastructure.config.data_ingestion_config import (
    DataIngestionConfig,
)


@inject
class AlembicMigrationService(MigrationServiceInterface):
    """
    Alembic-based implementation of migration service.

    This service handles database schema evolution using Alembic migrations.
    It provides a clean interface for managing TimescaleDB schema changes
    while following hexagonal architecture principles.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the migration service with configuration.

        Args:
            config: Data ingestion configuration containing database settings
        """
        self.config = config
        self.db_config = config.infrastructure.database
        self.logger = logging.getLogger(__name__)

        # Build connection string from injected config
        self.connection_string = self._build_connection_string()

        # Initialize Alembic configuration
        self.alembic_cfg = self._setup_alembic_config()

    def _build_connection_string(self) -> str:
        """
        Build PostgreSQL connection string from configuration.

        Returns:
            str: Complete PostgreSQL connection string
        """
        return (
            f"postgresql://{self.db_config.username}:{self.db_config.password}"
            f"@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        )

    def _setup_alembic_config(self) -> Config:
        """
        Setup Alembic configuration with proper paths and settings.

        Returns:
            Config: Configured Alembic Config object
        """
        # Path to alembic.ini relative to service root
        service_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        alembic_ini_path = os.path.join(service_root, "alembic.ini")

        if not os.path.exists(alembic_ini_path):
            self.logger.warning(f"alembic.ini not found at {alembic_ini_path}")
            # Create minimal config in memory for initial setup
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", "migrations")
        else:
            alembic_cfg = Config(alembic_ini_path)

        # Always override connection string from injected config
        alembic_cfg.set_main_option("sqlalchemy.url", self.connection_string)

        return alembic_cfg

    def migrate_to_latest(self) -> None:
        """
        Apply all pending migrations to the latest revision.

        Raises:
            MigrationError: If migration fails
        """
        try:
            self.logger.info("Applying migrations to latest revision...")
            command.upgrade(self.alembic_cfg, "head")
            self.logger.info("Successfully migrated to latest revision")
        except Exception as e:
            self.logger.error(f"Migration to latest failed: {str(e)}")
            raise MigrationError(f"Failed to migrate to latest: {str(e)}") from e

    def migrate_to_revision(self, revision: str) -> None:
        """
        Migrate to a specific revision.

        Args:
            revision: Target revision identifier

        Raises:
            MigrationError: If migration fails
        """
        try:
            self.logger.info(f"Migrating to revision: {revision}")
            command.upgrade(self.alembic_cfg, revision)
            self.logger.info(f"Successfully migrated to revision: {revision}")
        except Exception as e:
            self.logger.error(f"Migration to {revision} failed: {str(e)}")
            raise MigrationError(f"Failed to migrate to {revision}: {str(e)}") from e

    def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """
        Create a new migration file.

        Args:
            message: Description of the migration
            autogenerate: Whether to auto-detect schema changes

        Returns:
            str: The revision ID of the created migration

        Raises:
            MigrationError: If migration creation fails
        """
        try:
            self.logger.info(f"Creating migration: {message}")

            if autogenerate:
                revision = command.revision(
                    self.alembic_cfg, message=message, autogenerate=True
                )
            else:
                revision = command.revision(self.alembic_cfg, message=message)

            # Extract revision ID(s) from the returned object (list of Script or None)
            if isinstance(revision, list) and revision:
                revision_ids = [rev.revision for rev in revision if rev is not None]
                revision_id = revision_ids[0] if revision_ids else "unknown"
            else:
                revision_id = "unknown"
            self.logger.info(f"Created migration with revision ID: {revision_id}")
            return revision_id

        except Exception as e:
            self.logger.error(f"Failed to create migration: {str(e)}")
            raise MigrationError(f"Failed to create migration: {str(e)}") from e

    def get_current_revision(self) -> Optional[str]:
        """
        Get the current database revision.

        Returns:
            Optional[str]: Current revision ID, None if no migrations applied
        """
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            self.logger.error(f"Failed to get current revision: {str(e)}")
            return None

    def get_pending_migrations(self) -> List[str]:
        """
        Get list of pending migration revisions.

        Returns:
            List[str]: List of pending revision IDs
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()

            if current_rev is None:
                # No migrations applied yet, all are pending
                return [rev.revision for rev in script.walk_revisions()]

            pending = []
            for rev in script.walk_revisions():
                if rev.revision != current_rev:
                    pending.append(rev.revision)
                else:
                    break  # Found current revision, stop

            return pending

        except Exception as e:
            self.logger.error(f"Failed to get pending migrations: {str(e)}")
            return []

    def initialize_migration_repo(self) -> None:
        """
        Initialize Alembic migration repository.

        Raises:
            MigrationError: If initialization fails
        """
        try:
            self.logger.info("Initializing migration repository...")
            command.init(self.alembic_cfg, "migrations")
            self.logger.info("Migration repository initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize migration repo: {str(e)}")
            raise MigrationError(
                f"Failed to initialize migration repo: {str(e)}"
            ) from e

    def ensure_migrations_applied(self) -> None:
        """
        Ensure all migrations are applied on service startup.

        This method checks for pending migrations and applies them automatically.
        It should be called during service initialization.

        Raises:
            MigrationError: If migration check or application fails
        """
        try:
            pending = self.get_pending_migrations()

            if pending:
                self.logger.info(
                    f"Found {len(pending)} pending migrations, applying..."
                )
                self.migrate_to_latest()
            else:
                self.logger.info("Database is up to date, no migrations needed")

        except Exception as e:
            self.logger.error(f"Failed to ensure migrations: {str(e)}")
            raise MigrationError(f"Failed to ensure migrations: {str(e)}") from e
