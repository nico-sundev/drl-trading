"""Unit tests for MarketDataRepository.

Tests the repository functionality with proper mocking of session factory,
database queries, and mapper classes.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from drl_trading_adapter.adapter.database.repository.market_data_repository import MarketDataRepository
from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_core.core.model.data_availability_summary import DataAvailabilitySummary
from drl_trading_common.core.model.timeframe import Timeframe


@pytest.fixture
def mock_session_factory():
    """Create a mock session factory."""
    mock_factory = Mock()
    mock_session = Mock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    mock_factory.get_session.return_value = mock_session
    return mock_factory


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    mock = Mock()
    mock.__enter__ = Mock(return_value=mock)
    mock.__exit__ = Mock(return_value=None)
    return mock


@pytest.fixture
def mock_query():
    """Create a mock query object."""
    return Mock()


@pytest.fixture
def sample_entity():
    """Create a sample MarketDataEntity for testing."""
    entity = MarketDataEntity()
    entity.symbol = "AAPL"
    entity.timeframe = "1h"
    entity.timestamp = datetime(2024, 1, 15, 10, 0, 0)
    entity.open_price = 150.25
    entity.high_price = 152.75
    entity.low_price = 149.50
    entity.close_price = 151.80
    entity.volume = 1000000
    return entity


@pytest.fixture
def sample_model():
    """Create a sample MarketDataModel for testing."""
    return MarketDataModel(
        symbol="AAPL",
        timeframe=Timeframe.HOUR_1,
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        open_price=150.25,
        high_price=152.75,
        low_price=149.50,
        close_price=151.80,
        volume=1000000
    )


@pytest.fixture
def repository(mock_session_factory):
    """Create a MarketDataRepository instance with mocked dependencies."""
    return MarketDataRepository(mock_session_factory)


class TestMarketDataRepositoryRead:
    """Test suite for MarketDataRepository read operations."""

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.MarketDataMapper')
    def test_get_symbol_data_range_success(self, mock_mapper, repository, mock_session_factory, sample_entity, sample_model):
        """Test successful retrieval of symbol data range."""
        # Given
        symbol = "AAPL"
        timeframe = Timeframe.HOUR_1
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 31)

        # Get the session from the factory
        mock_session = mock_session_factory.get_session.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [sample_entity]
        mock_mapper.entity_to_model.return_value = sample_model

        # When
        result = repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

        # Then
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_model
        mock_mapper.entity_to_model.assert_called_once_with(sample_entity)

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.MarketDataMapper')
    def test_get_symbol_data_range_empty_result(self, mock_mapper, repository, mock_session_factory, mock_session):
        """Test retrieval when no data is found."""
        # Given
        symbol = "NONEXISTENT"
        timeframe = Timeframe.HOUR_1
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 31)

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        # When
        result = repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

        # Then
        assert isinstance(result, list)
        assert len(result) == 0
        mock_mapper.entity_to_model.assert_not_called()

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.MarketDataMapper')
    def test_get_symbol_data_range_database_error(self, mock_mapper, repository, mock_session_factory, mock_session):
        """Test handling of database errors during retrieval."""
        # Given
        symbol = "AAPL"
        timeframe = Timeframe.HOUR_1
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 31)

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = Exception("Database connection failed")

        # When & Then
        with pytest.raises(Exception, match="Database connection failed"):
            repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.MarketDataMapper')
    def test_get_latest_prices_success(self, mock_mapper, repository, mock_session_factory, mock_session, sample_entity, sample_model):
        """Test successful retrieval of latest market data."""
        # Given
        symbols = ["AAPL"]
        timeframe = Timeframe.HOUR_1

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.group_by.return_value.subquery.return_value = Mock()
        mock_session.query.return_value.join.return_value.all.return_value = [sample_entity]
        mock_mapper.entity_to_model.return_value = sample_model

        # When
        result = repository.get_latest_prices(symbols, timeframe)

        # Then
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_model
        mock_mapper.entity_to_model.assert_called_once_with(sample_entity)

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.DataAvailabilityMapper')
    def test_get_data_availability_success(self, mock_mapper, repository, mock_session_factory, mock_session):
        """Test successful retrieval of data availability for a specific symbol."""
        # Given
        symbol = "AAPL"
        timeframe = Timeframe.HOUR_1

        # Mock query result
        mock_result = Mock()
        # Use return_value instead of side_effect for repeated access
        mock_result.__getitem__ = Mock(return_value=100)  # Return 100 for any index access
        mock_result.__getitem__.side_effect = lambda x: {0: 100, 1: datetime(2024, 1, 1), 2: datetime(2024, 1, 31)}[x]

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_result

        # Mock mapper
        mock_availability = Mock(spec=DataAvailabilitySummary)
        mock_mapper.query_result_to_model.return_value = mock_availability

        # When
        result = repository.get_data_availability(symbol, timeframe)

        # Then
        assert isinstance(result, DataAvailabilitySummary)
        # Should be called once for the single symbol
        assert mock_mapper.query_result_to_model.call_count == 1

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.DataAvailabilityMapper')
    def test_get_data_availability_summary_success(self, mock_mapper, repository, mock_session_factory, mock_session):
        """Test successful retrieval of complete data availability summary."""
        # Given
        mock_result = Mock()
        mock_result.symbol = "AAPL"
        mock_result.timeframe = "1h"
        mock_result.__getitem__ = Mock(side_effect=[100, datetime(2024, 1, 1), datetime(2024, 1, 31)])  # [record_count, earliest, latest]

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.group_by.return_value.order_by.return_value.all.return_value = [mock_result]

        # Mock mapper
        mock_availability = Mock(spec=DataAvailabilitySummary)
        mock_mapper.query_result_to_model.return_value = mock_availability

        # When
        result = repository.get_data_availability_summary()

        # Then
        assert isinstance(result, list)
        assert len(result) == 1
        mock_mapper.query_result_to_model.assert_called_once()


class TestMarketDataRepositoryIntegration:
    """Test suite for MarketDataRepository integration scenarios."""

    @patch('drl_trading_adapter.adapter.database.repository.market_data_repository.MarketDataMapper')
    def test_session_context_manager_usage(self, mock_mapper, repository, mock_session_factory, mock_session):
        """Test that session context manager is properly used."""
        # Given
        symbol = "AAPL"
        timeframe = Timeframe.HOUR_1
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 31)

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        # When
        repository.get_symbol_data_range(symbol, timeframe, start_time, end_time)

        # Then
        mock_session_factory.get_session.assert_called_once()
        mock_session_factory.get_session.return_value.__enter__.assert_called_once()

    def test_repository_injection_pattern(self, mock_session_factory):
        """Test that repository follows proper dependency injection pattern."""
        # Given & When
        repository = MarketDataRepository(mock_session_factory)

        # Then
        assert repository.session_factory == mock_session_factory
        assert hasattr(repository, 'logger')
