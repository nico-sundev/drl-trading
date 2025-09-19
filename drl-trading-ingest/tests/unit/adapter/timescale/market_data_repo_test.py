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
        """Mock SQLAlchemy session factory."""
        return Mock()

    @pytest.fixture
    def repository(self, mock_session_factory):
        """Create repository instance with mocked dependencies."""
        return MarketDataRepo(mock_session_factory)

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
        mock_session = Mock()
        mock_session_factory.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.get_session.return_value.__exit__ = Mock(return_value=None)

        # When
        repository.save_market_data(symbol, timeframe, sample_dataframe)

        # Then
        mock_session_factory.get_session.assert_called_once()
        assert mock_session.merge.call_count == 2  # Two rows in DataFrame
        mock_session.commit.assert_called_once()

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

        # When
        repository.save_market_data(symbol, timeframe, empty_df)

        # Then
        mock_session_factory.get_session.assert_not_called()

    @patch('drl_trading_ingest.adapter.timescale.market_data_repo.func')
    def test_get_latest_timestamp_success(self, mock_func, repository, mock_session_factory):
        """Test successful latest timestamp retrieval."""
        # Given
        symbol = "AAPL"
        timeframe = "1h"
        expected_timestamp = datetime(2024, 1, 1, 12, 0)

        mock_session = Mock()
        mock_session_factory.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.get_session.return_value.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.scalar.return_value = expected_timestamp

        # When
        result = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert result == expected_timestamp.isoformat()
        mock_session.query.assert_called_once()

    @patch('drl_trading_ingest.adapter.timescale.market_data_repo.func')
    def test_get_latest_timestamp_no_data(self, mock_func, repository, mock_session_factory):
        """Test latest timestamp retrieval when no data exists."""
        # Given
        symbol = "NONEXISTENT"
        timeframe = "1h"

        mock_session = Mock()
        mock_session_factory.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.get_session.return_value.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.scalar.return_value = None

        # When
        result = repository.get_latest_timestamp(symbol, timeframe)

        # Then
        assert result is None
