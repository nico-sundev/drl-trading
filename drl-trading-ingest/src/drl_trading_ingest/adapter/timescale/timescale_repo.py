"""
TimescaleDB repository implementation for storing time series data.

This module provides the concrete implementation of TimescaleRepoInterface
for storing market data in TimescaleDB. It focuses solely on data operations,
with schema management delegated to the migration service.
"""

import logging
from typing import List, Optional

import psycopg2
import psycopg2.extras
from injector import inject
from pandas import DataFrame

from drl_trading_ingest.core.port.database_connection_interface import (
    DatabaseConnectionInterface,
)
from drl_trading_ingest.core.port.timescale_repo_interface import TimescaleRepoInterface

logger = logging.getLogger(__name__)


@inject
class TimescaleRepo(TimescaleRepoInterface):
    """
    TimescaleDB repository for storing time series data.

    This repository focuses solely on data operations, assuming that
    the database schema is properly managed by the migration service.
    It follows the single responsibility principle by separating
    data operations from schema management.
    """

    def __init__(self, connection_service: DatabaseConnectionInterface):
        """
        Initialize the repository with database connection service.

        Args:
            connection_service: Database connection interface for connection management
        """
        self.connection_service = connection_service
        self.logger = logging.getLogger(__name__)

    def store_timeseries_to_db(self, symbol: str, timeframe: str, df: DataFrame) -> None:
        """
        Store time series data to the unified market data table.

        This method assumes that the market_data table already exists and has been
        created through the migration system. It handles only data insertion with
        proper conflict resolution.

        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1H", "1D", "5M")
            df: DataFrame containing OHLCV data with required columns

        Raises:
            ValueError: If required columns are missing from the DataFrame
            DatabaseConnectionError: If database operation fails
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
            with self.connection_service.get_transaction() as cursor:
                # Use UPSERT to handle conflicts gracefully
                upsert_query = """
                    INSERT INTO market_data (
                        symbol, timeframe, timestamp,
                        open_price, high_price, low_price, close_price, volume
                    ) VALUES %s
                    ON CONFLICT (symbol, timeframe, timestamp)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        created_at = NOW()
                """

                # Prepare data tuples for batch insert
                data_tuples = self._prepare_data_tuples(symbol, timeframe, df)

                # Execute batch insert using psycopg2.extras.execute_values for efficiency
                psycopg2.extras.execute_values(
                    cursor,
                    upsert_query,
                    data_tuples,
                    template=None,
                    page_size=1000  # Process in batches for memory efficiency
                )

                rows_affected = cursor.rowcount
                self.logger.info(
                    f"Successfully stored {rows_affected} rows for {symbol}:{timeframe}"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to store data for {symbol}:{timeframe}: {str(e)}"
            )
            raise

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get the latest timestamp for a symbol/timeframe combination.

        This is useful for incremental data updates to avoid duplicates.

        Args:
            symbol: The trading symbol
            timeframe: The timeframe

        Returns:
            str: Latest timestamp as ISO string, or None if no data exists
        """
        try:
            with self.connection_service.get_connection() as connection:
                with connection.cursor() as cursor:
                    query = """
                        SELECT MAX(timestamp)
                        FROM market_data
                        WHERE symbol = %s AND timeframe = %s
                    """

                    cursor.execute(query, (symbol, timeframe))
                    result = cursor.fetchone()

                    if result and result[0]:
                        return result[0].isoformat()
                    return None

        except Exception as e:
            self.logger.error(
                f"Failed to get latest timestamp for {symbol}:{timeframe}: {str(e)}"
            )
            raise

    def _prepare_data_tuples(self, symbol: str, timeframe: str, df: DataFrame) -> List[tuple]:
        """
        Prepare data tuples for batch insertion.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            df: DataFrame with market data

        Returns:
            List[tuple]: Data tuples ready for insertion
        """
        data_tuples = []

        for _, row in df.iterrows():
            # Handle volume - default to 0 if not present
            volume = row.get('volume', 0)
            if volume is None or volume == '':
                volume = 0

            data_tuple = (
                symbol,
                timeframe,
                row['timestamp'],
                float(row['open_price']),
                float(row['high_price']),
                float(row['low_price']),
                float(row['close_price']),
                int(volume)
            )
            data_tuples.append(data_tuple)

        return data_tuples
