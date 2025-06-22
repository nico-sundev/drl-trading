"""Create unified market data table

Revision ID: 001_initial_tables
Revises:
Create Date: 2025-01-01 12:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '001_initial_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create unified market data table with TimescaleDB hypertable.

    This migration creates a single table for all market data with proper
    partitioning by symbol and timeframe for optimal performance.
    """
    # Create TimescaleDB extension if not exists
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # Create unified market data table
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open_price DOUBLE PRECISION NOT NULL,
            high_price DOUBLE PRECISION NOT NULL,
            low_price DOUBLE PRECISION NOT NULL,
            close_price DOUBLE PRECISION NOT NULL,
            volume BIGINT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, timeframe, timestamp)
        );
    """)

    # Convert to hypertable for time-series optimization
    op.execute("""
        SELECT create_hypertable('market_data', 'timestamp',
                                if_not_exists => TRUE);
    """)

    # Create composite index for symbol+timeframe queries (most common)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_time
        ON market_data (symbol, timeframe, timestamp DESC);
    """)

    # Create index for time-range queries across all symbols
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_desc
        ON market_data (timestamp DESC);
    """)

    # Create covering index for OHLC price queries (avoids table lookups)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_ohlc_covering
        ON market_data (symbol, timeframe, timestamp)
        INCLUDE (open_price, high_price, low_price, close_price, volume);
    """)

    # Create index for symbol-only queries (useful for portfolio-level analysis)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time
        ON market_data (symbol, timestamp DESC);
    """)


def downgrade() -> None:
    """
    Drop market data table and associated objects.

    This will remove the market data table and its TimescaleDB configuration.
    """
    op.execute("DROP TABLE IF EXISTS market_data CASCADE;")
