"""
Integration tests for Alembic migration service.

This module tests the complete migration service integration including
database connectivity, migration application, and error handling.
"""

from unittest.mock import Mock, patch

import pytest
from injector import Injector

from drl_trading_ingest.adapter.migration.alembic_migration_service import (
    AlembicMigrationService,
)
from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationError,
    MigrationServiceInterface,
)
from drl_trading_ingest.infrastructure.config.data_ingestion_config import (
    DataIngestionConfig,
)
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig
from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule


class TestAlembicMigrationServiceIntegration:
    """Integration tests for the Alembic migration service."""

    @pytest.fixture
    def mock_config(self) -> DataIngestionConfig:
        """Create a mock configuration for testing."""
        # Given
        config = Mock(spec=DataIngestionConfig)
        config.infrastructure = Mock()
        config.infrastructure.database = Mock()
        config.infrastructure.database.username = "test_user"
        config.infrastructure.database.password = "test_pass"
        config.infrastructure.database.host = "localhost"
        config.infrastructure.database.port = 5432
        config.infrastructure.database.database = "test_db"
        return config

    @pytest.fixture
    def migration_service(self, mock_config: DataIngestionConfig) -> AlembicMigrationService:
        """Create migration service instance for testing."""
        # Given
        return AlembicMigrationService(mock_config)

    def test_build_connection_string(self, migration_service: AlembicMigrationService) -> None:
        """Test that connection string is built correctly from config."""
        # Given
        # Migration service initialized with mock config

        # When
        connection_string = migration_service._build_connection_string()

        # Then
        expected = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert connection_string == expected

    def test_setup_alembic_config_without_ini_file(self, migration_service: AlembicMigrationService) -> None:
        """Test Alembic config setup when alembic.ini doesn't exist."""
        # Given
        # Migration service initialized (alembic.ini likely doesn't exist in test)

        # When
        alembic_cfg = migration_service._setup_alembic_config()

        # Then
        assert alembic_cfg is not None
        assert alembic_cfg.get_main_option("sqlalchemy.url") == migration_service.connection_string

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.command')
    def test_migrate_to_latest_success(self, mock_command, migration_service: AlembicMigrationService) -> None:
        """Test successful migration to latest revision."""
        # Given
        mock_command.upgrade = Mock()

        # When
        migration_service.migrate_to_latest()

        # Then
        mock_command.upgrade.assert_called_once_with(migration_service.alembic_cfg, "head")

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.command')
    def test_migrate_to_latest_failure(self, mock_command, migration_service: AlembicMigrationService) -> None:
        """Test migration failure handling."""
        # Given
        mock_command.upgrade.side_effect = Exception("Migration failed")

        # When & Then
        with pytest.raises(MigrationError, match="Failed to migrate to latest"):
            migration_service.migrate_to_latest()

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.command')
    def test_migrate_to_revision(self, mock_command, migration_service: AlembicMigrationService) -> None:
        """Test migration to specific revision."""
        # Given
        mock_command.upgrade = Mock()
        target_revision = "abc123"

        # When
        migration_service.migrate_to_revision(target_revision)

        # Then
        mock_command.upgrade.assert_called_once_with(migration_service.alembic_cfg, target_revision)

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.command')
    def test_create_migration_with_autogenerate(self, mock_command, migration_service: AlembicMigrationService) -> None:
        """Test migration creation with autogenerate enabled."""
        # Given
        from alembic.script.revision import Revision
        mock_revision = Mock(spec=Revision)
        mock_revision.revision = "def456"
        mock_command.revision.return_value = mock_revision



        # When
        revision_id = migration_service.create_migration("Test migration", autogenerate=True)

        # Then
        mock_command.revision.assert_called_once_with(
            migration_service.alembic_cfg,
            message="Test migration",
            autogenerate=True
        )
        assert revision_id == "def456"

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.create_engine')
    def test_get_current_revision_success(self, mock_create_engine, migration_service: AlembicMigrationService) -> None:
        """Test getting current revision successfully."""
        # Given
        mock_engine = Mock()
        mock_connection = Mock()
        mock_context = Mock()
        mock_context.get_current_revision.return_value = "current123"

        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_connection

        with patch('drl_trading_ingest.adapter.migration.alembic_migration_service.MigrationContext') as mock_migration_context:
            mock_migration_context.configure.return_value = mock_context

            # When
            current_rev = migration_service.get_current_revision()

            # Then
            assert current_rev == "current123"
            mock_connection.close.assert_called_once()

    @patch('drl_trading_ingest.adapter.migration.alembic_migration_service.create_engine')
    def test_get_current_revision_failure(self, mock_create_engine, migration_service: AlembicMigrationService) -> None:
        """Test handling of failure when getting current revision."""
        # Given
        mock_create_engine.side_effect = Exception("Database connection failed")

        # When
        current_rev = migration_service.get_current_revision()

        # Then
        assert current_rev is None

    def test_ensure_migrations_applied_with_pending(self, migration_service: AlembicMigrationService) -> None:
        """Test ensure migrations when there are pending migrations."""
        # Given
        with patch.object(migration_service, 'get_pending_migrations') as mock_get_pending, \
             patch.object(migration_service, 'migrate_to_latest') as mock_migrate:

            mock_get_pending.return_value = ["rev1", "rev2"]

            # When
            migration_service.ensure_migrations_applied()

            # Then
            mock_migrate.assert_called_once()

    def test_ensure_migrations_applied_no_pending(self, migration_service: AlembicMigrationService) -> None:
        """Test ensure migrations when no pending migrations exist."""
        # Given
        with patch.object(migration_service, 'get_pending_migrations') as mock_get_pending, \
             patch.object(migration_service, 'migrate_to_latest') as mock_migrate:

            mock_get_pending.return_value = []

            # When
            migration_service.ensure_migrations_applied()

            # Then
            mock_migrate.assert_not_called()


class TestDependencyInjectionIntegration:
    """Test migration service integration with the DI container."""

    def test_migration_service_injection(self) -> None:
        """Test that migration service can be injected from the DI container."""
        # Given
        # Mock the config loading to avoid file dependencies
        with patch('drl_trading_ingest.infrastructure.di.ingest_module.ServiceConfigLoader') as mock_loader:
            # Create a proper mock config that matches IngestConfig
            mock_config = Mock(spec=IngestConfig)
            mock_config.app_name = "drl-trading-ingest"
            mock_config.infrastructure = Mock()
            mock_config.infrastructure.database = Mock()
            mock_config.infrastructure.database.username = "test"
            mock_config.infrastructure.database.password = "test"
            mock_config.infrastructure.database.host = "localhost"
            mock_config.infrastructure.database.port = 5432
            mock_config.infrastructure.database.database = "test"
            mock_config.infrastructure.messaging = Mock()
            mock_config.infrastructure.messaging.host = "localhost:9092"

            mock_loader.load_config.return_value = mock_config

            # Mock environment variable
            with patch.dict('os.environ', {'SERVICE_CONFIG_PATH': '/mock/path'}):
                injector = Injector([IngestModule()])

                # When
                migration_service = injector.get(MigrationServiceInterface)

                # Then
                assert migration_service is not None
                assert isinstance(migration_service, AlembicMigrationService)
                assert migration_service.connection_string == "postgresql://test:test@localhost:5432/test"
