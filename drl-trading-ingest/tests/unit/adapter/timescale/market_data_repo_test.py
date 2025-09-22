"""
Unit tests for MarketDataRepo using SQLAlchemy ORM.

This test suite validates the refactored repository implementation
that uses MarketDataEntity and SQLAlchemy session factory.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd

from drl_trading_ingest.adapter.timescale.market_data_repo import MarketDataRepo


class TestMarketDataRepo:
    """Test suite for the modernized MarketDataRepo."""

    @pytest.fixture
    def mock_session_factory(self):
        """Mock SQLAlchemy session factory with proper context manager support."""
        mock_factory = Mock()
        mock_session = Mock()

        # Configure the context manager behavior
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        mock_factory.get_session.return_value = mock_context

        return mock_factory, mock_session

    @pytest.fixture
    def repository(self, mock_session_factory):
        """Create repository instance with mocked dependencies."""
        mock_factory, _ = mock_session_factory
        return MarketDataRepo(mock_factory)

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame with market data."""
        return pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)],
            'open_price': [100.0, 101.0],
            'high_price': [102.0, 103.0],
            'low_price': [99.0, 100.0],
            'close_price': [101.0, 102.0],
            'volume': [1000, 1500]
        })

    def test_save_market_data_success(self, repository, mock_session_factory, sample_dataframe):
        """Test successful market data saving with SQLAlchemy ORM."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # When
        repository.save_market_data(symbol, timeframe, sample_dataframe)

        # Then
        mock_factory.get_session.assert_called_once()
        # Verify that merge was called for each row in the DataFrame
        assert mock_session.merge.call_count == 2  # Two rows in DataFrame
        mock_session.commit.assert_called_once()

        # Verify that the entities created have correct data
        merge_calls = mock_session.merge.call_args_list
        assert len(merge_calls) == 2

        # Check first entity
        first_entity = merge_calls[0][0][0]
        assert first_entity.symbol == symbol
        assert first_entity.timeframe == timeframe
        assert first_entity.timestamp == datetime(2024, 1, 1, 10, 0)
        assert first_entity.open_price == 100.0
        assert first_entity.volume == 1000

    def test_save_market_data_missing_columns(self, repository, mock_session_factory):
        """Test error handling for missing required columns."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})

        # When & Then
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            repository.save_market_data(symbol, timeframe, invalid_df)

    def test_save_market_data_empty_dataframe(self, repository, mock_session_factory):
        """Test handling of empty DataFrame."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        empty_df = pd.DataFrame(columns=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price'])
        mock_factory, mock_session = mock_session_factory

        # When
        repository.save_market_data(symbol, timeframe, empty_df)

        # Then
        # Should not call get_session when DataFrame is empty
        mock_factory.get_session.assert_not_called()
        mock_session.merge.assert_not_called()
        mock_session.commit.assert_not_called()

    def test_save_market_data_volume_handling(self, repository, mock_session_factory):
        """Test volume handling with missing, None, and empty values."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # Test data with various volume scenarios
        test_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 12, 0)],
            'open_price': [100.0, 101.0, 102.0],
            'high_price': [102.0, 103.0, 104.0],
            'low_price': [99.0, 100.0, 101.0],
            'close_price': [101.0, 102.0, 103.0],
            'volume': [1000, None, '']  # Normal, None, and empty string
        })

        # When
        repository.save_market_data(symbol, timeframe, test_df)

        # Then
        assert mock_session.merge.call_count == 3
        merge_calls = mock_session.merge.call_args_list

        # Check volume handling
        assert merge_calls[0][0][0].volume == 1000  # Normal volume
        assert merge_calls[1][0][0].volume == 0     # None -> 0
        assert merge_calls[2][0][0].volume == 0     # Empty string -> 0

    def test_save_market_data_no_volume_column(self, repository, mock_session_factory):
        """Test handling of DataFrame without volume column."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # DataFrame without volume column
        no_volume_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'open_price': [100.0],
            'high_price': [102.0],
            'low_price': [99.0],
            'close_price': [101.0]
        })

        # When
        repository.save_market_data(symbol, timeframe, no_volume_df)

        # Then
        mock_session.merge.assert_called_once()
        merge_call = mock_session.merge.call_args_list[0]
        entity = merge_call[0][0]
        assert entity.volume == 0  # Should default to 0

    @patch('drl_trading_ingest.adapter.timescale.market_data_repo.func')
    def test_get_latest_timestamp_success(self, mock_func, repository, mock_session_factory):
        """Test successful latest timestamp retrieval."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        expected_timestamp = datetime(2024, 1, 1, 12, 0)
        mock_factory, mock_session = mock_session_factory

        # Mock the query chain
        mock_query = Mock()
        mock_filter = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.scalar.return_value = expected_timestamp

        # When
        result = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert result == expected_timestamp.isoformat()
        mock_factory.get_session.assert_called_once()
        mock_session.query.assert_called_once()
        mock_query.filter.assert_called_once()
        mock_filter.scalar.assert_called_once()

    @patch('drl_trading_ingest.adapter.timescale.market_data_repo.func')
    def test_get_latest_timestamp_no_data(self, mock_func, repository, mock_session_factory):
        """Test latest timestamp retrieval when no data exists."""
        # Given
        symbol = "NONEXISTENT"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # Mock the query chain to return None
        mock_query = Mock()
        mock_filter = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.scalar.return_value = None

        # When
        result = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert result is None
        mock_factory.get_session.assert_called_once()
        mock_session.query.assert_called_once()
        mock_query.filter.assert_called_once()
        mock_filter.scalar.assert_called_once()

    def test_save_market_data_database_error(self, repository, mock_session_factory, sample_dataframe):
        """Test error handling during database operations for save_market_data."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # Configure session to raise exception during commit
        mock_session.commit.side_effect = Exception("Database connection failed")

        # When & Then
        with pytest.raises(Exception, match="Database connection failed"):
            repository.save_market_data(symbol, timeframe, sample_dataframe)

    def test_get_latest_timestamp_database_error(self, repository, mock_session_factory):
        """Test error handling during database operations for get_latest_timestamp."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # Configure session to raise exception during query
        mock_session.query.side_effect = Exception("Database query failed")

        # When & Then
        with pytest.raises(Exception, match="Database query failed"):
            repository.get_latest_timestamp(symbol, timeframe)

    def test_save_market_data_type_conversion_error(self, repository, mock_session_factory):
        """Test error handling for data type conversion errors."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        mock_factory, mock_session = mock_session_factory

        # DataFrame with invalid data types
        invalid_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 10, 0)],
            'open_price': ['invalid_price'],  # String instead of float
            'high_price': [102.0],
            'low_price': [99.0],
            'close_price': [101.0],
            'volume': [1000]
        })

        # When & Then
        with pytest.raises(ValueError):
            repository.save_market_data(symbol, timeframe, invalid_df)
