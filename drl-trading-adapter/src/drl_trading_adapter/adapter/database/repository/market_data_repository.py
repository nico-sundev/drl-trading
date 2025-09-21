"""SQLAlchemy-based market data repository implementation.

This module provides Entity Framework-style repository implementation using
SQLAlchemy ORM for both read and write operations on market data.
"""

import logging
from datetime import datetime
from typing import List

from sqlalchemy import func, and_
from injector import inject

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_core.common.model.data_availability_summary import DataAvailabilitySummary
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_adapter.adapter.database.mapper import MarketDataMapper, DataAvailabilityMapper

logger = logging.getLogger(__name__)


@inject
class MarketDataRepository(MarketDataReaderPort):
    """SQLAlchemy-based market data repository.

    Provides Entity Framework-style data access using SQLAlchemy ORM.
    Implements the read-only port for shared access while also providing
    write operations for the ingest service.
    """

    def __init__(self, session_factory: SQLAlchemySessionFactory) -> None:
        """Initialize repository with SQLAlchemy session factory.

        Args:
            session_factory: SQLAlchemy session factory for database access
        """
        self.session_factory = session_factory
        self.logger = logging.getLogger(__name__)

    def _entity_to_model(self, entity: MarketDataEntity) -> MarketDataModel:
        """Convert MarketDataEntity to MarketDataModel."""
        return MarketDataMapper.entity_to_model(entity)

    def get_symbol_data_range_paginated(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        limit: int,
        offset: int = 0
    ) -> List[MarketDataModel]:
        """Retrieve market data for a symbol within the specified time range with pagination.

        Args:
            symbol: Trading symbol to fetch data for
            timeframe: Timeframe for the market data
            start_time: Start of the time range (inclusive)
            end_time: End of the time range (inclusive)
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)

        Returns:
            List of market data models within the specified range and pagination
        """
        with self.session_factory.get_session() as session:
            try:
                query = session.query(MarketDataEntity).filter(
                    and_(
                        MarketDataEntity.symbol == symbol,
                        MarketDataEntity.timeframe == timeframe.value,
                        MarketDataEntity.timestamp >= start_time,
                        MarketDataEntity.timestamp <= end_time
                    )
                ).order_by(MarketDataEntity.timestamp.asc()).limit(limit).offset(offset)

                entities = query.all()

                self.logger.debug(
                    f"Retrieved {len(entities)} records for {symbol} "
                    f"(timeframe: {timeframe.value}, limit: {limit}, offset: {offset})"
                )

                return [self._entity_to_model(entity) for entity in entities]

            except Exception as e:
                self.logger.error(f"Error fetching paginated market data: {e}")
                raise

    def get_symbol_data_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataModel]:
        """Retrieve market data for a symbol within the specified time range."""
        with self.session_factory.get_session() as session:
            try:
                query = session.query(MarketDataEntity).filter(
                    and_(
                        MarketDataEntity.symbol == symbol,
                        MarketDataEntity.timeframe == timeframe.value,
                        MarketDataEntity.timestamp >= start_time,
                        MarketDataEntity.timestamp <= end_time
                    )
                ).order_by(MarketDataEntity.timestamp.asc())

                entities = query.all()
                return [self._entity_to_model(entity) for entity in entities]

            except Exception as e:
                self.logger.error(f"Error fetching market data: {e}")
                raise

    def get_multiple_symbols_data_range(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataModel]:
        """Retrieve market data for multiple symbols within the specified time range."""
        if not symbols:
            return []

        with self.session_factory.get_session() as session:
            try:
                query = session.query(MarketDataEntity).filter(
                    and_(
                        MarketDataEntity.symbol.in_(symbols),
                        MarketDataEntity.timeframe == timeframe.value,
                        MarketDataEntity.timestamp >= start_time,
                        MarketDataEntity.timestamp <= end_time
                    )
                ).order_by(MarketDataEntity.symbol.asc(), MarketDataEntity.timestamp.asc())

                entities = query.all()
                return [self._entity_to_model(entity) for entity in entities]

            except Exception as e:
                self.logger.error(f"Error fetching market data for symbols: {e}")
                raise

    def get_latest_prices(
        self,
        symbols: List[str],
        timeframe: Timeframe
    ) -> List[MarketDataModel]:
        """Retrieve the latest market data record for each symbol."""
        if not symbols:
            return []

        with self.session_factory.get_session() as session:
            try:
                max_timestamp_subq = session.query(
                    MarketDataEntity.symbol,
                    func.max(MarketDataEntity.timestamp).label('max_timestamp')
                ).filter(
                    and_(
                        MarketDataEntity.symbol.in_(symbols),
                        MarketDataEntity.timeframe == timeframe.value
                    )
                ).group_by(MarketDataEntity.symbol).subquery()

                query = session.query(MarketDataEntity).join(
                    max_timestamp_subq,
                    and_(
                        MarketDataEntity.symbol == max_timestamp_subq.c.symbol,
                        MarketDataEntity.timestamp == max_timestamp_subq.c.max_timestamp,
                        MarketDataEntity.timeframe == timeframe.value
                    )
                )

                entities = query.all()
                return [self._entity_to_model(entity) for entity in entities]

            except Exception as e:
                self.logger.error(f"Error fetching latest market data: {e}")
                raise

    def get_data_availability(
        self,
        symbols: List[str],
        timeframe: Timeframe
    ) -> List[DataAvailabilitySummary]:
        """Get data availability summary for symbols and timeframe."""
        if not symbols:
            return []

        with self.session_factory.get_session() as session:
            try:
                availability_list = []

                for symbol in symbols:
                    query = session.query(
                        func.count(MarketDataEntity.timestamp).label('record_count'),
                        func.min(MarketDataEntity.timestamp).label('earliest'),
                        func.max(MarketDataEntity.timestamp).label('latest')
                    ).filter(
                        and_(
                            MarketDataEntity.symbol == symbol,
                            MarketDataEntity.timeframe == timeframe.value
                        )
                    )

                    result = query.first()

                    if result and result[0] > 0:  # record_count is the 1st column (index 0)
                        availability = DataAvailabilityMapper.query_result_to_model(
                            symbol=symbol,
                            timeframe=timeframe.value,
                            record_count=result[0],  # record_count
                            earliest_timestamp=result[1],  # earliest
                            latest_timestamp=result[2]  # latest
                        )
                        availability_list.append(availability)
                    else:
                        availability = DataAvailabilityMapper.query_result_to_model(
                            symbol=symbol,
                            timeframe=timeframe.value,
                            record_count=0,
                            earliest_timestamp=None,
                            latest_timestamp=None
                        )
                        availability_list.append(availability)

                return availability_list

            except Exception as e:
                self.logger.error(f"Error checking data availability: {e}")
                raise

    def get_symbol_available_timeframes(self, symbol: str) -> List[Timeframe]:
        """Get all available timeframes for a symbol."""
        with self.session_factory.get_session() as session:
            try:
                query = session.query(MarketDataEntity.timeframe).filter(
                    MarketDataEntity.symbol == symbol
                ).distinct()

                results = query.all()
                timeframes = [Timeframe(result.timeframe) for result in results]

                self.logger.debug(f"Found {len(timeframes)} timeframes for {symbol}")
                return timeframes

            except Exception as e:
                self.logger.error(f"Error getting timeframes for {symbol}: {e}")
                raise

    def get_data_availability_summary(self) -> List[DataAvailabilitySummary]:
        """Get summary of data availability across all symbols and timeframes."""
        with self.session_factory.get_session() as session:
            try:
                query = session.query(
                    MarketDataEntity.symbol,
                    MarketDataEntity.timeframe,
                    func.count(MarketDataEntity.timestamp).label('record_count'),
                    func.min(MarketDataEntity.timestamp).label('earliest'),
                    func.max(MarketDataEntity.timestamp).label('latest')
                ).group_by(
                    MarketDataEntity.symbol,
                    MarketDataEntity.timeframe
                ).order_by(
                    MarketDataEntity.symbol.asc(),
                    MarketDataEntity.timeframe.asc()
                )

                results = query.all()
                availability_list = []

                for result in results:
                    availability = DataAvailabilityMapper.query_result_to_model(
                        symbol=result.symbol,
                        timeframe=result.timeframe,
                        record_count=result[2],  # record_count is the 3rd column (index 2)
                        earliest_timestamp=result[3],  # earliest is the 4th column (index 3)
                        latest_timestamp=result[4]  # latest is the 5th column (index 4)
                    )
                    availability_list.append(availability)

                self.logger.debug(f"Generated {len(availability_list)} availability summaries")
                return availability_list

            except Exception as e:
                self.logger.error(f"Error getting data availability summary: {e}")
                raise
