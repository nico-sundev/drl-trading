"""
Integration tests for MarketDataRepo using testcontainers with real PostgreSQL.

This test suite validates the complete MarketDataRepo functionality
against a real PostgreSQL database using testcontainers for isolation.
"""

import pytest
import pandas as pd
from datetime import datetime
from testcontainers.postgres import PostgresContainer

from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity, Base
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_ingest.adapter.timescale.market_data_repo import MarketDataRepo
from drl_trading_common.config.infrastructure_config import DatabaseConfig


class TestMarketDataRepoIntegration:
    """Integration test suite for MarketDataRepo with real PostgreSQL."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Start PostgreSQL container for integration tests."""
        with PostgresContainer("postgres:15") as postgres:
            yield postgres

    @pytest.fixture(scope="class")
    def database_config(self, postgres_container):
        """Create database configuration from container."""
        return DatabaseConfig(
            host=postgres_container.get_container_host_ip(),
            port=postgres_container.get_exposed_port(5432),
            database=postgres_container.dbname,
            username=postgres_container.username,
            password=postgres_container.password
        )

    @pytest.fixture(scope="class")
    def session_factory(self, database_config):
        """Create session factory with test database."""
        return SQLAlchemySessionFactory(database_config)

    @pytest.fixture(scope="class", autouse=True)
    def setup_database_schema(self, session_factory):
        """Create database schema for tests."""
        # Create all tables using the engine from session factory
        Base.metadata.create_all(session_factory._engine)
        yield
        # Cleanup after tests
        Base.metadata.drop_all(session_factory._engine)

    @pytest.fixture
    def repository(self, session_factory):
        """Create repository instance for testing."""
        return MarketDataRepo(session_factory)

    @pytest.fixture(autouse=True)
    def clean_database(self, session_factory):
        """Clean database before each test."""
        with session_factory.get_session() as session:
            session.query(MarketDataEntity).delete()
            session.commit()
        yield

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data DataFrame for testing."""
        return pd.DataFrame({
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0)
            ],
            'open_price': [100.0, 101.5, 102.0],
            'high_price': [101.0, 102.5, 103.0],
            'low_price': [99.5, 101.0, 101.5],
            'close_price': [101.0, 102.0, 102.5],
            'volume': [1000, 1500, 2000]
        })

    def test_save_and_retrieve_market_data(self, repository, session_factory, sample_market_data):
        """Test complete save and retrieve cycle with real database."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"

        # When - Save data
        repository.save_market_data(symbol, timeframe, sample_market_data)

        # Then - Verify data was saved
        with session_factory.get_read_only_session() as session:
            saved_entities = session.query(MarketDataEntity).filter(
                MarketDataEntity.symbol == symbol,
                MarketDataEntity.timeframe == timeframe
            ).order_by(MarketDataEntity.timestamp).all()

            assert len(saved_entities) == 3

            # Verify first record
            first_entity = saved_entities[0]
            assert first_entity.symbol == symbol
            assert first_entity.timeframe == timeframe
            assert first_entity.timestamp == datetime(2024, 1, 1, 10, 0)
            assert first_entity.open_price == 100.0
            assert first_entity.high_price == 101.0
            assert first_entity.low_price == 99.5
            assert first_entity.close_price == 101.0
            assert first_entity.volume == 1000

            # Verify the created_at field is set (audit field)
            assert first_entity.created_at is not None

    def test_upsert_behavior(self, repository, session_factory, sample_market_data):
        """Test UPSERT behavior - update existing records."""
        # Given
        symbol = "TSLA"
        timeframe = "5m"

        # When - Save initial data
        repository.save_market_data(symbol, timeframe, sample_market_data)

        # Update the same timestamps with different prices
        updated_data = sample_market_data.copy()
        updated_data['close_price'] = [999.0, 998.0, 997.0]  # Different close prices

        # Save updated data
        repository.save_market_data(symbol, timeframe, updated_data)

        # Then - Verify data was updated, not duplicated
        with session_factory.get_read_only_session() as session:
            saved_entities = session.query(MarketDataEntity).filter(
                MarketDataEntity.symbol == symbol,
                MarketDataEntity.timeframe == timeframe
            ).order_by(MarketDataEntity.timestamp).all()

            assert len(saved_entities) == 3  # Still only 3 records

            # Verify prices were updated
            for entity, expected_close in zip(saved_entities, [999.0, 998.0, 997.0]):
                assert entity.close_price == expected_close

    def test_get_latest_timestamp(self, repository, sample_market_data):
        """Test retrieving latest timestamp for symbol/timeframe."""
        # Given
        symbol = "GOOGL"
        timeframe = "1d"

        # When - Save data
        repository.save_market_data(symbol, timeframe, sample_market_data)

        # Get latest timestamp
        latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert latest_timestamp is not None
        assert latest_timestamp == datetime(2024, 1, 1, 12, 0).isoformat()

    def test_get_latest_timestamp_no_data(self, repository):
        """Test latest timestamp for non-existent symbol."""
        # Given
        symbol = "NONEXISTENT"
        timeframe = "1h"

        # When
        latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert latest_timestamp is None

    def test_save_empty_dataframe(self, repository):
        """Test handling of empty DataFrame."""
        # Given
        symbol = "EMPTY"
        timeframe = "1h"
        empty_df = pd.DataFrame(columns=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price'])

        # When - Should not raise exception
        repository.save_market_data(symbol, timeframe, empty_df)

        # Then - No data should be saved
        latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)
        assert latest_timestamp is None

    def test_save_data_with_missing_volume(self, repository, session_factory):
        """Test handling of data without volume column."""
        # Given
        symbol = "NOVOL"
        timeframe = "1h"
        data_without_volume = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'open_price': [100.0],
            'high_price': [101.0],
            'low_price': [99.0],
            'close_price': [100.5]
            # No volume column - should default to 0
        })

        # When
        repository.save_market_data(symbol, timeframe, data_without_volume)

        # Then - Should save with volume = 0
        with session_factory.get_read_only_session() as session:
            entity = session.query(MarketDataEntity).filter(
                MarketDataEntity.symbol == symbol,
                MarketDataEntity.timeframe == timeframe
            ).first()

            assert entity is not None
            assert entity.volume == 0
            assert entity.open_price == 100.0
            assert entity.close_price == 100.5

    def test_save_data_with_null_volume_values(self, repository, session_factory):
        """Test handling of None and empty string volume values."""
        # Given
        symbol = "NULLVOL"
        timeframe = "1h"
        data_with_null_volumes = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)],
            'open_price': [100.0, 101.0],
            'high_price': [101.0, 102.0],
            'low_price': [99.0, 100.0],
            'close_price': [100.5, 101.5],
            'volume': [None, '']  # None and empty string should both become 0
        })

        # When
        repository.save_market_data(symbol, timeframe, data_with_null_volumes)

        # Then - Both records should have volume = 0
        with session_factory.get_read_only_session() as session:
            entities = session.query(MarketDataEntity).filter(
                MarketDataEntity.symbol == symbol,
                MarketDataEntity.timeframe == timeframe
            ).order_by(MarketDataEntity.timestamp).all()

            assert len(entities) == 2
            assert entities[0].volume == 0  # None -> 0
            assert entities[1].volume == 0  # '' -> 0

    def test_multiple_symbols_and_timeframes(self, repository, session_factory):
        """Test storing data for multiple symbols and timeframes."""
        # Given
        test_data = [
            ("AAPL", "1h", datetime(2024, 1, 1, 10, 0)),
            ("AAPL", "5m", datetime(2024, 1, 1, 10, 0)),
            ("GOOGL", "1h", datetime(2024, 1, 1, 10, 0)),
            ("GOOGL", "1d", datetime(2024, 1, 1, 0, 0)),
        ]

        # When - Save data for each combination
        for symbol, timeframe, timestamp in test_data:
            df = pd.DataFrame({
                'timestamp': [timestamp],
                'open_price': [100.0],
                'high_price': [101.0],
                'low_price': [99.0],
                'close_price': [100.5],
                'volume': [1000]
            })
            repository.save_market_data(symbol, timeframe, df)

        # Then - Verify all combinations exist
        with session_factory.get_read_only_session() as session:
            total_records = session.query(MarketDataEntity).count()
            assert total_records == 4

            # Verify each combination
            for symbol, timeframe, _ in test_data:
                entity = session.query(MarketDataEntity).filter(
                    MarketDataEntity.symbol == symbol,
                    MarketDataEntity.timeframe == timeframe
                ).first()
                assert entity is not None

    def test_database_constraint_handling(self, repository, session_factory):
        """Test that database constraints are properly handled."""
        # Given
        symbol = "CONSTRAINT_TEST"
        timeframe = "1h"

        # Create data with the same timestamp (should trigger upsert)
        df1 = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'open_price': [100.0],
            'high_price': [101.0],
            'low_price': [99.0],
            'close_price': [100.5],
            'volume': [1000]
        })

        df2 = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],  # Same timestamp
            'open_price': [110.0],  # Different values
            'high_price': [111.0],
            'low_price': [109.0],
            'close_price': [110.5],
            'volume': [2000]
        })

        # When - Save both datasets
        repository.save_market_data(symbol, timeframe, df1)
        repository.save_market_data(symbol, timeframe, df2)  # Should update, not fail

        # Then - Should have only one record with updated values
        with session_factory.get_read_only_session() as session:
            entities = session.query(MarketDataEntity).filter(
                MarketDataEntity.symbol == symbol,
                MarketDataEntity.timeframe == timeframe
            ).all()

            assert len(entities) == 1
            entity = entities[0]
            assert entity.open_price == 110.0  # Updated values
            assert entity.volume == 2000

    def test_save_data_with_invalid_columns(self, repository):
        """Test error handling for missing required columns."""
        # Given
        symbol = "INVALID"
        timeframe = "1h"
        invalid_df = pd.DataFrame({
            'wrong_column': [1, 2, 3],
            'another_wrong_column': ['a', 'b', 'c']
        })

        # When & Then - Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            repository.save_market_data(symbol, timeframe, invalid_df)
