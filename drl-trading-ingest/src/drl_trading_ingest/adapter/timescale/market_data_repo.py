"""
TimescaleDB repository implementation using SQLAlchemy ORM.

This module provides modern SQLAlchemy-based implementation for market data
operations, replacing the legacy raw SQL approach with proper entity mapping
and transaction management.
"""

import logging
from typing import Optional

from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from injector import inject
from pandas import DataFrame
from sqlalchemy import func

from drl_trading_ingest.core.port.market_data_repo_interface import (
    MarketDataRepoPort,
)

logger = logging.getLogger(__name__)


@inject
class MarketDataRepo(MarketDataRepoPort):
    """
    TimescaleDB repository using SQLAlchemy ORM for market data operations.

    This repository leverages the modern SQLAlchemy-based architecture
    with proper entity mapping, replacing the legacy raw SQL approach.
    It maintains the same interface while providing improved maintainability
    and testability through ORM abstraction.
    """

    def __init__(self, session_factory: SQLAlchemySessionFactory):
        """
        Initialize the repository with SQLAlchemy session factory.

        Args:
            session_factory: SQLAlchemy session factory for database access
        """
        self.session_factory = session_factory
        self.logger = logging.getLogger(__name__)

    def save_market_data(self, symbol: str, timeframe: str, df: DataFrame) -> None:
        """
        Store time series data using SQLAlchemy ORM with UPSERT semantics.

        Uses SQLAlchemy's merge() operation to handle conflicts gracefully,
        providing equivalent functionality to the previous raw SQL UPSERT.

        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1H", "1D", "5M")
            df: DataFrame containing OHLCV data with required columns

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        # Validate required columns exist
        required_columns = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        if df.empty:
            self.logger.warning(f"Empty DataFrame provided for {symbol}:{timeframe}")
            return

        try:
            with self.session_factory.get_session() as session:
                entities = []

                for _, row in df.iterrows():
                    # Handle volume - default to 0 if not present
                    volume = row.get('volume', 0)
                    if volume is None or volume == '':
                        volume = 0

                    entity = MarketDataEntity(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=row['timestamp'],
                        open_price=float(row['open_price']),
                        high_price=float(row['high_price']),
                        low_price=float(row['low_price']),
                        close_price=float(row['close_price']),
                        volume=int(volume)
                    )
                    entities.append(entity)

                # Use merge for UPSERT behavior - updates if exists, inserts if not
                for entity in entities:
                    session.merge(entity)

                session.commit()

                self.logger.info(
                    f"Successfully stored {len(entities)} rows for {symbol}:{timeframe}"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to store data for {symbol}:{timeframe}: {str(e)}"
            )
            raise

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get the latest timestamp for a symbol/timeframe combination using SQLAlchemy.

        This is useful for incremental data updates to avoid duplicates.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe

        Returns:
            str: Latest timestamp as ISO string, or None if no data exists
        """
        try:
            with self.session_factory.get_session() as session:
                result = session.query(func.max(MarketDataEntity.timestamp)).filter(
                    MarketDataEntity.symbol == symbol,
                    MarketDataEntity.timeframe == timeframe
                ).scalar()

                if result:
                    return result.isoformat()
                return None

        except Exception as e:
            self.logger.error(
                f"Failed to get latest timestamp for {symbol}:{timeframe}: {str(e)}"
            )
            raise
