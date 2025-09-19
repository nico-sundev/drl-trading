"""
Simplified unit tests for FeatureConfigRepository to verify basic functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from drl_trading_adapter.adapter.feature_config.feature_config_repository import FeatureConfigRepository
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity


def test_feature_config_reader_integration():
    """Simple integration test to verify reader can be instantiated."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)

    # When
    reader = FeatureConfigRepository(mock_session_factory)

    # Then
    assert reader is not None
    assert reader.session_factory == mock_session_factory


def test_feature_config_reader_validation():
    """Test validation logic in reader methods."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)
    reader = FeatureConfigRepository(mock_session_factory)

    # Mock session and query setup
    mock_session = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_session
    mock_session_factory.get_read_only_session.return_value = mock_context

    # Test config not found
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    # When / Then
    with pytest.raises(ValueError, match="Feature configuration version 'nonexistent' not found"):
        reader.get_config("nonexistent")


def test_feature_config_reader_exists_check():
    """Test config existence functionality."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)
    reader = FeatureConfigRepository(mock_session_factory)

    # Mock session
    mock_session = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_session
    mock_session_factory.get_read_only_session.return_value = mock_context

    # Test case: config exists
    mock_session.query.return_value.filter.return_value.first.return_value = FeatureConfigEntity(
        hash="test_hash", semver="1.0.0", created_at=datetime(2024, 1, 1), feature_definitions=[]
    )

    # When
    result = reader.is_config_existing("test_hash")

    # Then
    assert result is True

    # Test case: config does not exist
    mock_session.query.return_value.filter.return_value.first.return_value = None

    # When
    result = reader.is_config_existing("nonexistent")

    # Then
    assert result is False
