"""
Migration service interface for database schema management.

This module defines the interface for handling database migrations in the trading ingestion service.
Following hexagonal architecture principles, this interface serves as a port that can be implemented
by various migration providers (adapters).
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class MigrationError(Exception):
    """Exception raised for migration-related errors."""
    pass


class MigrationServiceInterface(ABC):
    """
    Interface for database migration operations.

    This interface defines the contract for managing database schema evolution,
    including applying migrations, creating new migrations, and querying migration status.
    Implementations should handle the underlying migration framework details.
    """

    @abstractmethod
    def migrate_to_latest(self) -> None:
        """
        Apply all pending migrations to bring database to latest schema.

        This method should apply all pending migrations in the correct order,
        ensuring the database schema is up-to-date with the latest version.

        Raises:
            MigrationError: If migration fails for any reason
        """
        pass

    @abstractmethod
    def migrate_to_revision(self, revision: str) -> None:
        """
        Migrate database to a specific revision.

        This allows for precise control over database schema version,
        supporting both upgrades and downgrades to specific revisions.

        Args:
            revision: Target revision identifier (e.g., revision hash or tag)

        Raises:
            MigrationError: If migration fails or revision is invalid
        """
        pass

    @abstractmethod
    def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """
        Create a new migration file.

        Generates a new migration file with the specified message.
        If autogenerate is True, the migration framework should attempt
        to detect schema changes automatically.

        Args:
            message: Descriptive message for the migration
            autogenerate: Whether to auto-detect schema changes

        Returns:
            str: The revision ID of the created migration

        Raises:
            MigrationError: If migration creation fails
        """
        pass

    @abstractmethod
    def get_current_revision(self) -> Optional[str]:
        """
        Get the current database revision.

        Returns the revision identifier of the currently applied migration.
        This helps in understanding the current state of the database schema.

        Returns:
            Optional[str]: Current revision ID, None if no migrations applied yet
        """
        pass

    @abstractmethod
    def get_pending_migrations(self) -> List[str]:
        """
        Get list of pending migration revisions.

        Returns a list of migration revisions that haven't been applied yet.
        This is useful for understanding what changes would be applied.

        Returns:
            List[str]: List of pending revision IDs in application order
        """
        pass

    @abstractmethod
    def initialize_migration_repo(self) -> None:
        """
        Initialize migration repository structure.

        Creates the necessary directory structure and configuration files
        for the migration framework. This should be called once during
        initial setup of the service.

        Raises:
            MigrationError: If initialization fails
        """
        pass

    @abstractmethod
    def ensure_migrations_applied(self) -> None:
        """
        Ensure all migrations are applied on service startup.

        This method should be called during service initialization to ensure
        the database schema is up-to-date. It's a convenience method that
        combines checking for pending migrations and applying them if needed.

        Raises:
            MigrationError: If migration check or application fails
        """
        pass
