"""
Simple unit test for FeatureConfigWriter to verify basic functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from drl_trading_training.adapter.feature_config.feature_config_writer import FeatureConfigWriter
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


def test_feature_config_writer_integration():
    """Simple integration test to verify writer can be instantiated."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)

    # When
    writer = FeatureConfigWriter(mock_session_factory)

    # Then
    assert writer is not None
    assert writer.session_factory == mock_session_factory


def test_feature_config_writer_validation():
    """Test validation logic in writer methods."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)
    writer = FeatureConfigWriter(mock_session_factory)

    # Test missing semver
    config_missing_semver = FeatureConfigVersionInfo(
        hash="test_hash",
        semver="",  # Empty semver
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        feature_definitions=[{"name": "test"}]
    )

    # When / Then
    with pytest.raises(ValueError, match="Configuration must have both semver and hash"):
        writer.save_config(config_missing_semver)


def test_feature_config_writer_save_success():
    """Test successful config save."""
    # Given
    mock_session_factory = Mock(spec=SQLAlchemySessionFactory)
    writer = FeatureConfigWriter(mock_session_factory)

    # Mock session setup
    mock_session = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_session
    mock_session_factory.get_session.return_value = mock_context

    # Mock entity with hash
    mock_entity = MagicMock()
    mock_entity.hash = "test_hash_123"
    mock_session.merge.return_value = mock_entity

    config = FeatureConfigVersionInfo(
        hash="test_hash_123",
        semver="1.0.0",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        feature_definitions=[{"name": "test_feature"}],
        description="Test config"
    )

    # When
    result_hash = writer.save_config(config)

    # Then
    assert result_hash == "test_hash_123"
    mock_session.merge.assert_called_once()
    mock_session.commit.assert_called_once()
