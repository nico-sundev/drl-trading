"""
SQLAlchemy session factory for database operations.

This module provides SQLAlchemy session management with proper transaction
handling and connection management optimized for TimescaleDB operations.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from injector import inject

from drl_trading_common.config.infrastructure_config import DatabaseConfig

logger = logging.getLogger(__name__)


class SessionFactoryError(Exception):
    """Exception raised for SQLAlchemy session factory related errors."""
    pass


@inject
class SQLAlchemySessionFactory:
    """
    SQLAlchemy session factory for modern ORM-based database operations.

    This factory provides SQLAlchemy ORM sessions with proper transaction
    management and connection handling optimized for TimescaleDB.
    """

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize session factory with database configuration.

        Args:
            db_config: Database configuration from infrastructure config
        """
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)

        # Create SQLAlchemy engine
        self._engine = self._create_engine()
        self._session_maker = sessionmaker(bind=self._engine)

    def _create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with proper configuration.

        Returns:
            Engine: Configured SQLAlchemy engine
        """
        connection_url = (
            f"postgresql://{self.db_config.username}:{self.db_config.password}"
            f"@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        )

        # Use NullPool to avoid connection conflicts with psycopg2 pool
        # SQLAlchemy will create connections as needed without pooling
        engine = create_engine(
            connection_url,
            poolclass=NullPool,  # No connection pooling - handled by psycopg2
            echo=False,  # Set to True for SQL debugging
            future=True  # Use SQLAlchemy 2.0 style
        )

        self.logger.info("SQLAlchemy engine created successfully")
        return engine

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get SQLAlchemy ORM session with automatic transaction management.

        Provides session with automatic commit on success and rollback on error.

        Yields:
            Session: SQLAlchemy ORM session

        Raises:
            SessionFactoryError: If session creation or transaction fails
        """
        session = None
        try:
            session = self._session_maker()

            yield session

            # Commit on successful completion
            session.commit()
            self.logger.debug("SQLAlchemy session committed successfully")

        except Exception as e:
            if session:
                session.rollback()
                self.logger.error(f"SQLAlchemy session rolled back due to error: {str(e)}")
            raise SessionFactoryError(f"Session error: {str(e)}") from e
        finally:
            if session:
                session.close()

    @contextmanager
    def get_read_only_session(self) -> Generator[Session, None, None]:
        """
        Get read-only SQLAlchemy session for query operations.

        Optimized for read operations with automatic rollback (no commits).

        Yields:
            Session: Read-only SQLAlchemy session

        Raises:
            SessionFactoryError: If session creation fails
        """
        session = None
        try:
            session = self._session_maker()

            yield session

            # Always rollback for read-only sessions
            session.rollback()

        except Exception as e:
            if session:
                session.rollback()
            self.logger.error(f"Read-only session error: {str(e)}")
            raise SessionFactoryError(f"Read-only session error: {str(e)}") from e
        finally:
            if session:
                session.close()

    def close_engine(self) -> None:
        """
        Close SQLAlchemy engine and all connections.

        Should be called during application shutdown.
        """
        try:
            if self._engine:
                self._engine.dispose()
                self.logger.info("SQLAlchemy engine disposed")
        except Exception as e:
            self.logger.error(f"Error disposing SQLAlchemy engine: {str(e)}")

    def __del__(self) -> None:
        """Cleanup engine on garbage collection."""
        self.close_engine()
