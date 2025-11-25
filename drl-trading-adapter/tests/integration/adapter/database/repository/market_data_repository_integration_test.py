"""
Integration tests for MarketDataRepository using testcontainers with real PostgreSQL.

This test suite validates the complete market data repository functionality
against a real PostgreSQL database using testcontainers for isolation.
"""

import pytest
from datetime import datetime, timedelta, timezone
from testcontainers.postgres import PostgresContainer

from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity, Base
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.database.repository.market_data_repository import MarketDataRepository
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_core.core.model.data_availability_summary import DataAvailabilitySummary


class TestMarketDataRepositoryIntegration:
    """Integration test suite for MarketDataRepository with real PostgreSQL."""

    @pytest.fixture(scope="function")
    def postgres_container(self):
        """Start PostgreSQL container for integration tests."""
        with PostgresContainer("postgres:15") as postgres:
            yield postgres

    @pytest.fixture(scope="function")
    def database_config(self, postgres_container):
        """Create database configuration from container."""
        return DatabaseConfig(
            host=postgres_container.get_container_host_ip(),
            port=postgres_container.get_exposed_port(5432),
            database=postgres_container.dbname,
            username=postgres_container.username,
            password=postgres_container.password
        )

    @pytest.fixture(scope="function")
    def session_factory(self, database_config):
        """Create session factory with test database."""
        return SQLAlchemySessionFactory(database_config)

    @pytest.fixture(scope="function", autouse=True)
    def setup_database_schema(self, session_factory):
        """Create database schema for tests."""
        # Create all tables
        Base.metadata.create_all(session_factory._engine)
        yield
        # Cleanup is handled by container disposal

    @pytest.fixture
    def repository(self, session_factory):
        """Create MarketDataRepository instance for testing."""
        return MarketDataRepository(session_factory)

    @pytest.fixture(scope="function")
    def sample_market_data_entities(self, session_factory):
        """Create sample market data entities in the database."""
        # Each test gets a fresh database container, so we can use the same timestamps
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        entities = []

        # Create test data for multiple symbols and timeframes
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        timeframes = [Timeframe.MINUTE_1, Timeframe.MINUTE_5, Timeframe.HOUR_1]

        for i, symbol in enumerate(symbols):
            for j, timeframe in enumerate(timeframes):
                for k in range(5):  # 5 records per symbol/timeframe
                    timestamp = base_time + timedelta(minutes=k * (j + 1))
                    base_price = 1.1000 + (i * 0.1) + (k * 0.0001)

                    entity = MarketDataEntity(
                        symbol=symbol,
                        timeframe=timeframe.value,
                        timestamp=timestamp,
                        open_price=base_price,
                        high_price=base_price + 0.0005,
                        low_price=base_price - 0.0003,
                        close_price=base_price + 0.0002,
                        volume=1000 + (k * 100)
                    )
                    entities.append(entity)

        # Insert entities into database
        with session_factory.get_session() as session:
            session.add_all(entities)
            session.commit()

        return entities

    def test_get_symbol_data_range_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of symbol data within time range."""
        # Given
        symbol = "EURUSD"
        timeframe = Timeframe.MINUTE_1
        start_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc)

        # When
        result = repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

        # Then
        assert len(result) > 0
        assert all(isinstance(model, MarketDataModel) for model in result)
        assert all(model.symbol == symbol for model in result)
        assert all(model.timeframe == timeframe for model in result)
        assert all(start_time <= model.timestamp <= end_time for model in result)
        # Verify ordering by timestamp
        timestamps = [model.timestamp for model in result]
        assert timestamps == sorted(timestamps)

    def test_get_symbol_data_range_empty_result(self, repository, sample_market_data_entities):
        """Test retrieval with time range that has no data."""
        # Given
        symbol = "EURUSD"
        timeframe = Timeframe.MINUTE_1
        start_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)  # Future date
        end_time = datetime(2025, 1, 1, 10, 10, 0, tzinfo=timezone.utc)

        # When
        result = repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

        # Then
        assert len(result) == 0

    def test_get_multiple_symbols_data_range_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of multiple symbols data."""
        # Given
        symbols = ["EURUSD", "GBPUSD"]
        timeframe = Timeframe.MINUTE_1
        start_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc)

        # When
        result = repository.get_multiple_symbols_data_range(symbols, timeframe, start_time, end_time)

        # Then
        assert len(result) > 0
        assert all(isinstance(model, MarketDataModel) for model in result)
        assert all(model.symbol in symbols for model in result)
        assert all(model.timeframe == timeframe for model in result)
        # Verify data for both symbols exists
        result_symbols = set(model.symbol for model in result)
        assert "EURUSD" in result_symbols
        assert "GBPUSD" in result_symbols

    def test_get_multiple_symbols_data_range_empty_symbols(self, repository):
        """Test retrieval with empty symbols list."""
        # Given
        symbols = []
        timeframe = Timeframe.MINUTE_1
        start_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc)

        # When
        result = repository.get_multiple_symbols_data_range(symbols, timeframe, start_time, end_time)

        # Then
        assert len(result) == 0

    def test_get_latest_prices_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of latest prices for symbols."""
        # Given
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        timeframe = Timeframe.MINUTE_1

        # When
        result = repository.get_latest_prices(symbols, timeframe)

        # Then
        assert len(result) == len(symbols)
        assert all(isinstance(model, MarketDataModel) for model in result)
        result_symbols = [model.symbol for model in result]
        for symbol in symbols:
            assert symbol in result_symbols

        # Verify we get the latest timestamp for each symbol
        for symbol in symbols:
            symbol_data = [model for model in result if model.symbol == symbol]
            assert len(symbol_data) == 1

    def test_get_latest_prices_empty_symbols(self, repository):
        """Test retrieval of latest prices with empty symbols list."""
        # Given
        symbols = []
        timeframe = Timeframe.MINUTE_1

        # When
        result = repository.get_latest_prices(symbols, timeframe)

        # Then
        assert len(result) == 0

    def test_get_data_availability_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of data availability summary."""
        # Given
        symbol = "EURUSD"
        timeframe = Timeframe.MINUTE_1

        # When
        result = repository.get_data_availability(symbol, timeframe)

        # Then
        assert isinstance(result, DataAvailabilitySummary)
        assert result.symbol == symbol
        assert result.timeframe == timeframe
        assert result.record_count > 0
        assert result.earliest_timestamp is not None
        assert result.latest_timestamp is not None
        assert result.earliest_timestamp <= result.latest_timestamp

    def test_get_data_availability_no_data(self, repository):
        """Test data availability for symbol with no data."""
        # Given
        symbol = "NONEXISTENT"
        timeframe = Timeframe.MINUTE_1

        # When
        result = repository.get_data_availability(symbol, timeframe)

        # Then
        assert isinstance(result, DataAvailabilitySummary)
        assert result.symbol == symbol
        assert result.record_count == 0
        assert result.earliest_timestamp is None
        assert result.latest_timestamp is None

    def test_get_symbol_available_timeframes_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of available timeframes for a symbol."""
        # Given
        symbol = "EURUSD"

        # When
        result = repository.get_symbol_available_timeframes(symbol)

        # Then
        assert len(result) > 0
        assert all(isinstance(tf, Timeframe) for tf in result)
        # Should contain at least the timeframes we inserted
        timeframe_values = [tf.value for tf in result]
        assert Timeframe.MINUTE_1.value in timeframe_values

    def test_get_symbol_available_timeframes_no_data(self, repository):
        """Test retrieval of timeframes for symbol with no data."""
        # Given
        symbol = "NONEXISTENT"

        # When
        result = repository.get_symbol_available_timeframes(symbol)

        # Then
        assert len(result) == 0

    def test_get_data_availability_summary_success(self, repository, sample_market_data_entities):
        """Test successful retrieval of complete data availability summary."""
        # Given
        # sample_market_data_entities fixture provides test data

        # When
        result = repository.get_data_availability_summary()

        # Then
        assert len(result) > 0

        # Should have data for multiple symbols and timeframes
        symbols = set(availability.symbol for availability in result)
        timeframes = set(availability.timeframe for availability in result)

        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert "USDJPY" in symbols

        assert Timeframe.MINUTE_1 in timeframes
        assert Timeframe.MINUTE_5 in timeframes
        assert Timeframe.HOUR_1 in timeframes

        # All records should have positive counts and valid timestamps
        for availability in result:
            assert availability.record_count > 0
            assert availability.earliest_timestamp is not None
            assert availability.latest_timestamp is not None
            assert availability.earliest_timestamp <= availability.latest_timestamp

    def test_repository_implements_port_interface(self, repository):
        """Test that repository properly implements the MarketDataReaderPort interface."""
        # Given
        from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort

        # When/Then
        assert isinstance(repository, MarketDataReaderPort)

        # Verify all required methods are implemented
        assert hasattr(repository, 'get_symbol_data_range')
        assert hasattr(repository, 'get_multiple_symbols_data_range')
        assert hasattr(repository, 'get_latest_prices')
        assert hasattr(repository, 'get_data_availability')
        assert hasattr(repository, 'get_symbol_available_timeframes')
        assert hasattr(repository, 'get_data_availability_summary')

    def test_session_factory_integration(self, repository, session_factory):
        """Test that repository properly integrates with session factory."""
        # Given
        # Repository should be initialized with session factory

        # When/Then
        assert repository.session_factory is session_factory

        # Test that we can get a session from the factory
        with session_factory.get_session() as session:
            assert session is not None
            # Verify we can query the database
            count = session.query(MarketDataEntity).count()
            assert count >= 0  # Should work without error
