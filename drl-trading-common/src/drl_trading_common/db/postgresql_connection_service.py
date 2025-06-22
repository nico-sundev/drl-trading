"""
PostgreSQL connection service implementation with connection pooling.

This module provides a production-ready implementation of database connection
management using psycopg2 connection pooling for optimal performance.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from drl_trading_common.db.database_connection_interface import DatabaseConnectionError, DatabaseConnectionInterface
import psycopg2
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from injector import inject
from psycopg2 import pool

logger = logging.getLogger(__name__)


@inject
class PostgreSQLConnectionService(DatabaseConnectionInterface):
    """
    PostgreSQL connection service with connection pooling.

    Provides efficient connection management using psycopg2 SimpleConnectionPool
    with proper resource cleanup and transaction management.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize connection service with configuration.

        Args:
            config: Data ingestion configuration containing database settings
        """
        self.db_config = config
        self.logger = logging.getLogger(__name__)

        # Build connection string
        self.connection_string = self._build_connection_string()

        # Initialize connection pool
        self._pool = self._create_connection_pool()

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

    def _create_connection_pool(self) -> pool.SimpleConnectionPool:
        """
        Create and initialize connection pool.

        Returns:
            pool.SimpleConnectionPool: Configured connection pool

        Raises:
            DatabaseConnectionError: If pool creation fails
        """
        try:
            # Create connection pool with reasonable defaults
            connection_pool = pool.SimpleConnectionPool(
                minconn=1,  # Minimum connections
                maxconn=20,  # Maximum connections
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
            )

            self.logger.info("Database connection pool initialized successfully")
            return connection_pool

        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {str(e)}")
            raise DatabaseConnectionError(f"Pool creation failed: {str(e)}") from e

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a database connection from the pool.

        Yields:
            psycopg2.connection: Database connection from pool

        Raises:
            DatabaseConnectionError: If connection cannot be obtained
        """
        connection = None
        try:
            connection = self._pool.getconn()
            if connection is None:
                raise DatabaseConnectionError("Failed to obtain connection from pool")

            yield connection

        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database connection error: {str(e)}")
            raise DatabaseConnectionError(f"Connection error: {str(e)}") from e
        finally:
            if connection:
                self._pool.putconn(connection)

    @contextmanager
    def get_transaction(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """
        Get a database transaction with automatic commit/rollback.

        Yields:
            psycopg2.cursor: Database cursor within transaction

        Raises:
            DatabaseConnectionError: If transaction cannot be started
        """
        with self.get_connection() as connection:
            cursor = None
            try:
                cursor = connection.cursor()

                # Begin transaction (autocommit is off by default)
                yield cursor

                # Commit on successful completion
                connection.commit()
                self.logger.debug("Transaction committed successfully")

            except Exception as e:
                # Rollback on any exception
                connection.rollback()
                self.logger.error(f"Transaction rolled back due to error: {str(e)}")
                raise
            finally:
                if cursor:
                    cursor.close()

    def close_all_connections(self) -> None:
        """
        Close all connections in the pool.

        Should be called during application shutdown.
        """
        try:
            if self._pool:
                self._pool.closeall()
                self.logger.info("All database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")

    def __del__(self) -> None:
        """Cleanup connections on garbage collection."""
        self.close_all_connections()
