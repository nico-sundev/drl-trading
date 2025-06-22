"""
Database connection service interface for TimescaleDB operations.

This module defines the interface for database connection management,
following dependency inversion principle and enabling proper resource management.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

import psycopg2


class DatabaseConnectionInterface(ABC):
    """
    Interface for database connection management.

    Provides connection pooling, transaction management, and resource cleanup
    following the repository pattern and clean architecture principles.
    """

    @abstractmethod
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a database connection from the pool.

        This context manager ensures proper connection cleanup and
        automatic rollback on exceptions.

        Yields:
            psycopg2.connection: Database connection from pool

        Raises:
            DatabaseConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    @contextmanager
    def get_transaction(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """
        Get a database transaction with automatic commit/rollback.

        This context manager handles transaction lifecycle:
        - Begins transaction
        - Commits on success
        - Rolls back on exception
        - Ensures connection cleanup

        Yields:
            psycopg2.cursor: Database cursor within transaction

        Raises:
            DatabaseConnectionError: If transaction cannot be started
        """
        pass

    @abstractmethod
    def close_all_connections(self) -> None:
        """
        Close all connections in the pool.

        Should be called during application shutdown to ensure
        proper resource cleanup.
        """
        pass


class DatabaseConnectionError(Exception):
    """Exception raised for database connection-related errors."""
    pass
