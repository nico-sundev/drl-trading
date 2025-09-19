"""
SQLAlchemy entity for market data table.

This module defines the MarketData entity that maps to the TimescaleDB market_data table
using SQLAlchemy ORM. The entity follows the existing database schema created by
the ingest service migrations.
"""

from datetime import datetime

from sqlalchemy import BigInteger, Column, Float, String, Index, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MarketDataEntity(Base):
    """
    SQLAlchemy entity for market data time series.

    Maps to the market_data table in TimescaleDB with proper column types
    and constraints matching the migration schema.
    """

    __tablename__ = 'market_data'

    # Primary key components (composite)
    symbol = Column(String(20), primary_key=True, nullable=False)
    timeframe = Column(String(10), primary_key=True, nullable=False)
    timestamp = Column(DateTime, primary_key=True, nullable=False)

    # OHLCV data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False, default=0)

    # Audit fields
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)

    # Define indexes to match migration schema
    __table_args__ = (
        # Composite index for symbol+timeframe queries (most common)
        Index('idx_market_data_symbol_timeframe_time', 'symbol', 'timeframe', 'timestamp'),

        # Index for time-range queries across all symbols
        Index('idx_market_data_timestamp_desc', 'timestamp'),

        # Covering index for OHLC price queries
        Index('idx_market_data_ohlc_covering', 'symbol', 'timeframe', 'timestamp'),

        # Index for symbol-only queries
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<MarketDataEntity(symbol='{self.symbol}', timeframe='{self.timeframe}', "
            f"timestamp='{self.timestamp}', close_price={self.close_price})>"
        )

    def to_dict(self) -> dict:
        """Convert entity to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MarketDataEntity':
        """Create entity from dictionary."""
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            timestamp=data['timestamp'],
            open_price=float(data['open_price']),
            high_price=float(data['high_price']),
            low_price=float(data['low_price']),
            close_price=float(data['close_price']),
            volume=int(data.get('volume', 0))
        )
