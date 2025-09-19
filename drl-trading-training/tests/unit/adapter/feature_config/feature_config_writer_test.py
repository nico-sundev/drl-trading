"""
Unit tests for FeatureConfigWriter.

Tests the write adapter functionality for feature configuration storage,
validating SQLAlchemy ORM operations and error handling.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from drl_trading_training.adapter.feature_config.feature_config_writer import FeatureConfigWriter
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory, SessionFactoryError
from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class TestFeatureConfigWriter:
    """Unit test suite for FeatureConfigWriter."""

    @pytest.fixture
    def mock_session_factory(self):
        """Mock session factory for testing."""
        return Mock(spec=SQLAlchemySessionFactory)

    @pytest.fixture
    def mock_session(self):
        """Mock session for testing."""
        session = Mock(spec=Session)
        return session

    @pytest.fixture
    def writer(self, mock_session_factory):
        """Create FeatureConfigWriter instance for testing."""
        return FeatureConfigWriter(mock_session_factory)

    @pytest.fixture
    def sample_config(self):
        """Sample FeatureConfigVersionInfo for testing."""
        return FeatureConfigVersionInfo(
            hash="abc123def456",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[
                {"name": "rsi_14", "type": "technical_indicator"},
                {"name": "sma_20", "type": "moving_average"}
            ],
            description="Test configuration"
        )

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

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_success(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity):
        """Test successful config save operation."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity

        # When
        result_hash = writer.save_config(sample_config)

        # Then
        assert result_hash == "abc123def456"
        mock_mapper.to_entity.assert_called_once_with(sample_config)
        mock_session.merge.assert_called_once_with(sample_entity)
        mock_session.commit.assert_called_once()

    def test_save_config_validation_missing_semver(self, writer, sample_config):
        """Test validation when semver is missing."""
        # Given
        sample_config.semver = ""

        # When / Then
        with pytest.raises(ValueError, match="Configuration must have both semver and hash"):
            writer.save_config(sample_config)

    def test_save_config_validation_missing_hash(self, writer, sample_config):
        """Test validation when hash is missing."""
        # Given
        sample_config.hash = ""

        # When / Then
        with pytest.raises(ValueError, match="Configuration must have both semver and hash"):
            writer.save_config(sample_config)

    def test_save_config_validation_none_semver(self, writer, sample_config):
        """Test validation when semver is None."""
        # Given
        sample_config.semver = None

        # When / Then
        with pytest.raises(ValueError, match="Configuration must have both semver and hash"):
            writer.save_config(sample_config)

    def test_save_config_validation_none_hash(self, writer, sample_config):
        """Test validation when hash is None."""
        # Given
        sample_config.hash = None

        # When / Then
        with pytest.raises(ValueError, match="Configuration must have both semver and hash"):
            writer.save_config(sample_config)

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_mapper_error(self, mock_mapper, writer, mock_session_factory, sample_config):
        """Test save operation when mapper throws error."""
        # Given
        mock_mapper.to_entity.side_effect = ValueError("Mapper validation failed")

        # When / Then
        with pytest.raises(ValueError, match="Mapper validation failed"):
            writer.save_config(sample_config)

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_database_error(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity):
        """Test save operation with database error."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.side_effect = Exception("Database connection failed")

        # When / Then
        with pytest.raises(SessionFactoryError, match="Failed to save config version 1.0.0"):
            writer.save_config(sample_config)

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_commit_error(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity):
        """Test save operation with commit error."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity
        mock_session.commit.side_effect = Exception("Commit failed")

        # When / Then
        with pytest.raises(SessionFactoryError, match="Failed to save config version 1.0.0"):
            writer.save_config(sample_config)

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_upsert_behavior(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity):
        """Test UPSERT behavior using session.merge()."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity

        # When
        writer.save_config(sample_config)

        # Then
        # Verify merge is used (which provides UPSERT semantics)
        mock_session.merge.assert_called_once_with(sample_entity)
        # Verify add is not called (merge handles both insert and update)
        mock_session.add.assert_not_called()

    def test_session_context_manager_usage(self, writer, mock_session_factory, sample_config):
        """Test that session context manager is used properly."""
        # Given
        mock_context = MagicMock()
        mock_session_factory.get_session.return_value = mock_context
        mock_context.__enter__.return_value.merge.side_effect = Exception("Test error")

        # When / Then
        with pytest.raises(SessionFactoryError):
            writer.save_config(sample_config)

        # Verify context manager was used
        mock_session_factory.get_session.assert_called_once()
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_logging_on_success(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity, caplog):
        """Test that successful operations are logged."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity

        # When
        writer.save_config(sample_config)

        # Then
        assert "Successfully saved config version 1.0.0 (hash: abc123def456)" in caplog.text

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_logging_on_error(self, mock_mapper, writer, mock_session_factory, mock_session, sample_config, sample_entity, caplog):
        """Test that errors are logged properly."""
        # Given
        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.side_effect = Exception("Test database error")

        # When / Then
        with pytest.raises(SessionFactoryError):
            writer.save_config(sample_config)

        # Verify error was logged
        assert "Failed to save config version 1.0.0" in caplog.text
        assert "Test database error" in caplog.text

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_with_none_description(self, mock_mapper, writer, mock_session_factory, mock_session, sample_entity):
        """Test saving config with None description."""
        # Given
        config_with_none_desc = FeatureConfigVersionInfo(
            hash="test_hash",
            semver="2.0.0",
            created_at=datetime(2024, 2, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}],
            description=None
        )

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity

        # When
        result_hash = writer.save_config(config_with_none_desc)

        # Then
        assert result_hash == "abc123def456"
        mock_mapper.to_entity.assert_called_once_with(config_with_none_desc)

    @patch('drl_trading_training.adapter.feature_config.feature_config_writer.FeatureConfigMapper')
    def test_save_config_with_complex_feature_definitions(self, mock_mapper, writer, mock_session_factory, mock_session, sample_entity):
        """Test saving config with complex feature definitions."""
        # Given
        complex_config = FeatureConfigVersionInfo(
            hash="complex_hash",
            semver="3.0.0",
            created_at=datetime(2024, 3, 1, 12, 0, 0),
            feature_definitions=[
                {
                    "name": "composite_indicator",
                    "type": "custom",
                    "components": [
                        {"name": "rsi", "weight": 0.3},
                        {"name": "macd", "weight": 0.7}
                    ],
                    "parameters": {
                        "lookback": 20,
                        "threshold": 0.5
                    }
                }
            ],
            description="Complex feature configuration"
        )

        mock_session_factory.get_session.return_value.__enter__.return_value = mock_session
        mock_mapper.to_entity.return_value = sample_entity
        mock_session.merge.return_value = sample_entity

        # When
        result_hash = writer.save_config(complex_config)

        # Then
        assert result_hash == "abc123def456"
        mock_mapper.to_entity.assert_called_once_with(complex_config)
