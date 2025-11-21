"""
Unit tests for FeatureConfigRepository.

Tests the read adapter functionality for feature configuration storage,
validating SQLAlchemy ORM operations and error handling.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from drl_trading_adapter.adapter.database.repository.feature_config_repository import FeatureConfigRepository
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory, SessionFactoryError
from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class TestFeatureConfigRepository:
    """Unit test suite for FeatureConfigRepository."""

    @pytest.fixture
    def mock_session_factory(self):
        """Mock session factory for testing."""
        mock_factory = Mock(spec=SQLAlchemySessionFactory)

        # Create a proper context manager mock
        mock_context = MagicMock()
        mock_factory.get_read_only_session.return_value = mock_context

        return mock_factory

    @pytest.fixture
    def mock_session(self):
        """Mock session for testing."""
        session = Mock(spec=Session)
        return session

    @pytest.fixture
    def reader(self, mock_session_factory):
        """Create FeatureConfigRepository instance for testing."""
        return FeatureConfigRepository(mock_session_factory)

    @pytest.fixture
    def sample_entity(self):
        """Sample FeatureConfigEntity for testing."""
        return FeatureConfigEntity(
            hash="abc123def456",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[
                {"name": "rsi_14", "type": "technical_indicator"},
                {"name": "sma_20", "type": "moving_average"}
            ],
            description="Test configuration"
        )

    def test_get_config_by_hash_success(self, reader, mock_session_factory, mock_session, sample_entity):
        """Test successful config retrieval by hash."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = sample_entity

        # When
        result = reader.get_config("abc123def456")

        # Then
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.hash == "abc123def456"
        assert result.semver == "1.0.0"
        assert result.description == "Test configuration"
        mock_session_factory.get_read_only_session.assert_called_once()

    def test_get_config_by_semver_success(self, reader, mock_session_factory, mock_session, sample_entity):
        """Test successful config retrieval by semver."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = sample_entity

        # When
        result = reader.get_config("1.0.0")

        # Then
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.semver == "1.0.0"
        assert result.hash == "abc123def456"

    def test_get_config_session_factory_error_with_not_found(self, reader, mock_session_factory):
        """Test config retrieval with SessionFactoryError containing 'not found'."""
        # Given
        mock_session_factory.get_read_only_session.side_effect = SessionFactoryError("Feature configuration version 'test' not found")

        # When / Then
        with pytest.raises(ValueError, match="Feature configuration version 'test' not found"):
            reader.get_config("test")

    def test_get_config_generic_error_with_logging(self, reader, mock_session_factory, mock_session, caplog):
        """Test config retrieval with generic exception and verify logging."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = RuntimeError("Unexpected runtime error")

        # When / Then
        with pytest.raises(SessionFactoryError, match="Failed to retrieve config version 'test'"):
            reader.get_config("test")

        # Verify error was logged
        assert "Failed to retrieve config version 'test'" in caplog.text
        assert "Unexpected runtime error" in caplog.text

    def test_is_config_existing_true(self, reader, mock_session_factory, mock_session, sample_entity):
        """Test config existence check when config exists."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = sample_entity

        # When
        result = reader.is_config_existing("abc123def456")

        # Then
        assert result is True

    def test_is_config_existing_false(self, reader, mock_session_factory, mock_session):
        """Test config existence check when config does not exist."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # When
        result = reader.is_config_existing("nonexistent")

        # Then
        assert result is False

    def test_is_config_existing_database_error(self, reader, mock_session_factory, mock_session):
        """Test config existence check with database error."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = Exception("Database error")

        # When / Then
        with pytest.raises(SessionFactoryError, match="Failed to check config existence for version 'test'"):
            reader.is_config_existing("test")

    def test_get_latest_config_by_semver_prefix_success(self, reader, mock_session_factory, mock_session, sample_entity):
        """Test successful retrieval of latest config by semver prefix."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = sample_entity

        # When
        result = reader.get_latest_config_by_semver_prefix("1.0")

        # Then
        assert result is not None
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.semver == "1.0.0"

    def test_get_latest_config_by_semver_prefix_not_found(self, reader, mock_session_factory, mock_session):
        """Test semver prefix retrieval when no matching config found."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        # When
        result = reader.get_latest_config_by_semver_prefix("99.99")

        # Then
        assert result is None

    def test_get_latest_config_by_semver_prefix_database_error(self, reader, mock_session_factory, mock_session):
        """Test semver prefix retrieval with database error."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = Exception("Database error")

        # When / Then
        with pytest.raises(SessionFactoryError, match="Failed to get latest config for prefix 'test'"):
            reader.get_latest_config_by_semver_prefix("test")

    @patch('drl_trading_adapter.adapter.database.repository.feature_config_repository.FeatureConfigMapper')
    def test_mapper_called_correctly(self, mock_mapper, reader, mock_session_factory, mock_session, sample_entity):
        """Test that mapper is called correctly during conversion."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = sample_entity

        expected_domain_model = FeatureConfigVersionInfo(
            hash="abc123def456",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}],
            description="Test"
        )
        mock_mapper.to_domain_model.return_value = expected_domain_model

        # When
        result = reader.get_config("test")

        # Then
        mock_mapper.to_domain_model.assert_called_once_with(sample_entity)
        assert result == expected_domain_model

    def test_session_context_manager_usage(self, reader, mock_session_factory):
        """Test that session context manager is used properly."""
        # Given
        mock_context = MagicMock()
        mock_session_factory.get_read_only_session.return_value = mock_context
        mock_context.__enter__.return_value.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        # When / Then
        with pytest.raises(ValueError):  # Config not found
            reader.get_config("test")

        # Verify context manager was used
        mock_session_factory.get_read_only_session.assert_called_once()
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    def test_logging_on_error(self, reader, mock_session_factory, mock_session, caplog):
        """Test that errors are logged properly."""
        # Given
        mock_session_factory.get_read_only_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = Exception("Test database error")

        # When / Then
        with pytest.raises(SessionFactoryError):
            reader.get_config("test")

        # Verify error was logged
        assert "Failed to retrieve config version 'test'" in caplog.text
        assert "Test database error" in caplog.text
