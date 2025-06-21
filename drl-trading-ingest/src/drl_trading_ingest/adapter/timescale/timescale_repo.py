
import psycopg2
from injector import inject

from drl_trading_ingest.core.timescale_repo_interface import TimescaleRepoInterface
from drl_trading_ingest.infrastructure.config.data_ingestion_config import (
    DataIngestionConfig,
)

DB_CONFIG = {
    "dbname": "marketdata",
    "user": "postgres",
    "password": "postgres",
    "host": "timescaledb",
    "port": 5432,
}

@inject
class TimescaleRepo(TimescaleRepoInterface):

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.db_config = config.infrastructure.database

    def store_timeseries_to_db(symbol, df):
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {symbol} (
                timestamp TIMESTAMPTZ PRIMARY KEY,
                price DOUBLE PRECISION
            );
        """)
        for _, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {symbol} (timestamp, price)
                VALUES (%s, %s)
                ON CONFLICT (timestamp) DO NOTHING;
            """, (row['timestamp'], row['price']))
        conn.commit()
        cursor.close()
        conn.close()
