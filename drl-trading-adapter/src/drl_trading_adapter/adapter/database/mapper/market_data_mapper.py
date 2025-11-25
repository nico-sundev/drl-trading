"""
Entity-to-Model mappers for market data domain.

This module provides mapping functions to convert between database entities
and business domain models, ensuring clean separation of persistence and
business concerns.
"""

from datetime import datetime, timezone
from typing import Any

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_core.core.model.data_availability_summary import DataAvailabilitySummary
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity


class MarketDataMapper:
    """
    Mapper for converting between MarketDataEntity and MarketDataModel.

    Handles the transformation between database persistence representation
    and business domain representation of market data.
    """

    @staticmethod
    def entity_to_model(entity: MarketDataEntity) -> MarketDataModel:
        """
        Convert MarketDataEntity to MarketDataModel.

        Normalizes timestamps to Python's built-in datetime.timezone.utc to ensure
        consistency across the application, regardless of which timezone implementation
        the database driver uses (pytz.UTC, dateutil.tz.UTC, etc.).

        Args:
            entity: Database entity from SQLAlchemy query

        Returns:
            MarketDataModel: Business domain representation

        Raises:
            ValueError: If entity contains invalid data
        """
        if not entity:
            raise ValueError("Entity cannot be None")

        # Check for required fields that cannot be None
        required_fields = ['symbol', 'timeframe', 'timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        none_fields = [field for field in required_fields if getattr(entity, field, None) is None]
        if none_fields:
            raise ValueError("Entity has None values in required fields: " + ", ".join(none_fields))

        try:
            # Normalize timestamp to Python's built-in datetime.timezone.utc
            # This handles cases where SQLAlchemy returns pytz.UTC or dateutil.tz.UTC
            normalized_timestamp = entity.timestamp.astimezone(timezone.utc) if entity.timestamp.tzinfo else entity.timestamp.replace(tzinfo=timezone.utc)

            return MarketDataModel(
                symbol=entity.symbol,
                timestamp=normalized_timestamp,
                timeframe=Timeframe(entity.timeframe),
                open_price=entity.open_price,
                high_price=entity.high_price,
                low_price=entity.low_price,
                close_price=entity.close_price,
                volume=entity.volume
            )
        except Exception as e:
            raise ValueError(f"Failed to convert entity to model: {e}") from e

    @staticmethod
    def model_to_entity(model: MarketDataModel) -> MarketDataEntity:
        """
        Convert MarketDataModel to MarketDataEntity.

        Args:
            model: Business domain model

        Returns:
            MarketDataEntity: Database entity representation

        Raises:
            ValueError: If model contains invalid data
        """
        if not model:
            raise ValueError("Model cannot be None")

        if not isinstance(model, MarketDataModel):
            raise ValueError("Model must be an instance of MarketDataModel")

        try:
            entity = MarketDataEntity()
            entity.symbol = model.symbol
            entity.timestamp = model.timestamp
            entity.timeframe = model.timeframe.value
            entity.open_price = model.open_price
            entity.high_price = model.high_price
            entity.low_price = model.low_price
            entity.close_price = model.close_price
            entity.volume = model.volume
            return entity
        except Exception as e:
            raise ValueError(f"Failed to convert model to entity: {e}") from e
class DataAvailabilityMapper:
    """
    Mapper for converting query results to DataAvailabilitySummary models.

    Handles transformation of aggregated database query results into
    business domain models for data availability reporting.
    """

    @staticmethod
    def query_result_to_model(
        symbol: str,
        timeframe: str,
        record_count: int,
        earliest_timestamp: datetime | None,
        latest_timestamp: datetime | None
    ) -> DataAvailabilitySummary:
        """
        Convert aggregated query result to DataAvailabilitySummary model.

        Normalizes timestamps to Python's built-in datetime.timezone.utc to ensure
        consistency across the application, regardless of which timezone implementation
        the database driver uses.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string value
            record_count: Number of records found
            earliest_timestamp: Earliest timestamp or None if no data
            latest_timestamp: Latest timestamp or None if no data

        Returns:
            DataAvailabilitySummary: Business domain representation

        Raises:
            ValueError: If required parameters are invalid
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if record_count < 0:
            raise ValueError("Record count cannot be negative")

        try:
            # Normalize timestamps to Python's built-in datetime.timezone.utc
            normalized_earliest = None
            if earliest_timestamp is not None:
                normalized_earliest = earliest_timestamp.astimezone(timezone.utc) if earliest_timestamp.tzinfo else earliest_timestamp.replace(tzinfo=timezone.utc)

            normalized_latest = None
            if latest_timestamp is not None:
                normalized_latest = latest_timestamp.astimezone(timezone.utc) if latest_timestamp.tzinfo else latest_timestamp.replace(tzinfo=timezone.utc)

            return DataAvailabilitySummary(
                symbol=symbol,
                timeframe=Timeframe(timeframe),
                record_count=record_count,
                earliest_timestamp=normalized_earliest,
                latest_timestamp=normalized_latest
            )
        except Exception as e:
            raise ValueError(f"Failed to create DataAvailabilitySummary: {e}") from e

    @staticmethod
    def query_result_row_to_model(result_row: Any) -> DataAvailabilitySummary:
        """
        Convert SQLAlchemy query result row to DataAvailabilitySummary model.

        Args:
            result_row: SQLAlchemy result row with attributes:
                       symbol, timeframe, count, earliest, latest

        Returns:
            DataAvailabilitySummary: Business domain representation

        Raises:
            ValueError: If result row is invalid or missing required attributes
        """
        if not result_row:
            raise ValueError("Result row cannot be None")

        try:
            return DataAvailabilityMapper.query_result_to_model(
                symbol=result_row.symbol,
                timeframe=result_row.timeframe,
                record_count=result_row.count,
                earliest_timestamp=result_row.earliest,
                latest_timestamp=result_row.latest
            )
        except AttributeError as e:
            raise ValueError(f"Result row missing required attributes: {e}") from e
