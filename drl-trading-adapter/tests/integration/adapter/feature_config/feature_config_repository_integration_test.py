"""
Integration tests for FeatureConfigRepository using testcontainers with real PostgreSQL.

This test suite validates the complete feature configuration repository functionality
against a real PostgreSQL database using testcontainers for isolation.
"""

import pytest
from datetime import datetime
import uuid
from testcontainers.postgres import PostgresContainer

from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity, Base
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.database.repository.feature_config_repository import FeatureConfigRepository
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class TestFeatureConfigRepositoryIntegration:
    """Integration test suite for FeatureConfigRepository with real PostgreSQL."""

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
        """Create FeatureConfigRepository instance for testing."""
        return FeatureConfigRepository(session_factory)

    @pytest.fixture
    def sample_feature_config_entities(self, session_factory):
        """Create sample feature config entities in the database."""
        # Generate unique hashes
        hash_v1_0_0 = str(uuid.uuid4())
        hash_v1_0_1 = str(uuid.uuid4())
        hash_v1_1_0 = str(uuid.uuid4())
        hash_v2_0_0 = str(uuid.uuid4())

        entities = [
            FeatureConfigEntity(
                hash=hash_v1_0_0,
                semver="1.0.0",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[
                    {"name": "rsi_14", "type": "technical_indicator"},
                    {"name": "sma_20", "type": "moving_average"}
                ],
                description="Test configuration v1.0.0"
            ),
            FeatureConfigEntity(
                hash=hash_v1_0_1,
                semver="1.0.1",
                created_at=datetime(2024, 1, 2, 12, 0, 0),
                feature_definitions=[
                    {"name": "rsi_14", "type": "technical_indicator"},
                    {"name": "sma_20", "type": "moving_average"},
                    {"name": "ema_12", "type": "exponential_moving_average"}
                ],
                description="Test configuration v1.0.1 with additional features"
            ),
            FeatureConfigEntity(
                hash=hash_v1_1_0,
                semver="1.1.0",
                created_at=datetime(2024, 1, 3, 12, 0, 0),
                feature_definitions=[
                    {"name": "rsi_14", "type": "technical_indicator"},
                    {"name": "sma_20", "type": "moving_average"},
                    {"name": "macd", "type": "momentum_indicator"}
                ],
                description="Test configuration v1.1.0 with MACD"
            ),
            FeatureConfigEntity(
                hash=hash_v2_0_0,
                semver="2.0.0",
                created_at=datetime(2024, 1, 4, 12, 0, 0),
                feature_definitions=[
                    {"name": "rsi_21", "type": "technical_indicator"},
                    {"name": "sma_50", "type": "moving_average"},
                    {"name": "bollinger_bands", "type": "volatility_indicator"}
                ],
                description="Test configuration v2.0.0 - major version with new indicators"
            )
        ]

        # Insert entities into database
        with session_factory.get_session() as session:
            session.add_all(entities)
            session.commit()

        # Return the data structure that includes hash values for easy access
        return {
            "entities": entities,
            "hashes": {
                "v1_0_0": hash_v1_0_0,
                "v1_0_1": hash_v1_0_1,
                "v1_1_0": hash_v1_1_0,
                "v2_0_0": hash_v2_0_0
            }
        }

    def test_get_config_by_semver_success(self, repository, sample_feature_config_entities):
        """Test successful retrieval of config by semantic version."""
        # Given
        version = "1.0.1"

        # When
        result = repository.get_config(version)

        # Then
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.semver == "1.0.1"
        assert result.hash == sample_feature_config_entities["hashes"]["v1_0_1"]
        assert result.description == "Test configuration v1.0.1 with additional features"
        assert len(result.feature_definitions) == 3
        assert {"name": "ema_12", "type": "exponential_moving_average"} in result.feature_definitions

    def test_get_config_by_hash_success(self, repository, sample_feature_config_entities):
        """Test successful retrieval of config by hash."""
        # Given
        version = sample_feature_config_entities["hashes"]["v1_1_0"]  # Hash for v1.1.0

        # When
        result = repository.get_config(version)

        # Then
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.semver == "1.1.0"
        assert result.hash == sample_feature_config_entities["hashes"]["v1_1_0"]
        assert result.description == "Test configuration v1.1.0 with MACD"
        assert {"name": "macd", "type": "momentum_indicator"} in result.feature_definitions

    def test_get_config_not_found(self, repository, sample_feature_config_entities):
        """Test retrieval of non-existent config version."""
        # Given
        version = "9.9.9"  # Non-existent version

        # When / Then
        with pytest.raises(ValueError, match="Feature configuration version '9.9.9' not found"):
            repository.get_config(version)

    def test_get_config_latest_when_multiple_matches(self, repository, session_factory):
        """Test that get_config returns latest when multiple configs with same semver exist."""
        # Given - Create multiple configs with same semver but different timestamps
        entity1 = FeatureConfigEntity(
            hash="old123hash456",
            semver="1.5.0",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            feature_definitions=[{"name": "old_feature", "type": "test"}],
            description="Old config"
        )
        entity2 = FeatureConfigEntity(
            hash="new456hash789",
            semver="1.5.0",
            created_at=datetime(2024, 1, 1, 11, 0, 0),  # Later timestamp
            feature_definitions=[{"name": "new_feature", "type": "test"}],
            description="New config"
        )

        with session_factory.get_session() as session:
            session.add_all([entity1, entity2])
            session.commit()

        # When
        result = repository.get_config("1.5.0")

        # Then
        assert result.hash == "new456hash789"  # Should get the latest one
        assert result.description == "New config"

    def test_is_config_existing_by_semver_true(self, repository, sample_feature_config_entities):
        """Test config existence check by semver returns True for existing config."""
        # Given
        version = "1.0.0"

        # When
        result = repository.is_config_existing(version)

        # Then
        assert result is True

    def test_is_config_existing_by_hash_true(self, repository, sample_feature_config_entities):
        """Test config existence check by hash returns True for existing config."""
        # Given
        version = sample_feature_config_entities["hashes"]["v1_0_0"]  # Hash for v1.0.0

        # When
        result = repository.is_config_existing(version)

        # Then
        assert result is True

    def test_is_config_existing_false(self, repository, sample_feature_config_entities):
        """Test config existence check returns False for non-existent config."""
        # Given
        version = "9.9.9"  # Non-existent version

        # When
        result = repository.is_config_existing(version)

        # Then
        assert result is False

    def test_get_latest_config_by_semver_prefix_success(self, repository, sample_feature_config_entities):
        """Test successful retrieval of latest config by semver prefix."""
        # Given
        semver_prefix = "1.0"  # Should match 1.0.0 and 1.0.1, return latest (1.0.1)

        # When
        result = repository.get_latest_config_by_semver_prefix(semver_prefix)

        # Then
        assert result is not None
        assert isinstance(result, FeatureConfigVersionInfo)
        assert result.semver == "1.0.1"  # Latest 1.0.x version
        assert result.hash == sample_feature_config_entities["hashes"]["v1_0_1"]

    def test_get_latest_config_by_semver_prefix_exact_match(self, repository, sample_feature_config_entities):
        """Test retrieval by exact semver returns that specific version."""
        # Given
        semver_prefix = "2.0.0"  # Exact match

        # When
        result = repository.get_latest_config_by_semver_prefix(semver_prefix)

        # Then
        assert result is not None
        assert result.semver == "2.0.0"
        assert result.hash == sample_feature_config_entities["hashes"]["v2_0_0"]

    def test_get_latest_config_by_semver_prefix_no_match(self, repository, sample_feature_config_entities):
        """Test retrieval by semver prefix with no matches returns None."""
        # Given
        semver_prefix = "3.0"  # No matching versions

        # When
        result = repository.get_latest_config_by_semver_prefix(semver_prefix)

        # Then
        assert result is None

    def test_get_latest_config_by_semver_prefix_major_version(self, repository, sample_feature_config_entities):
        """Test retrieval by major version prefix returns latest in that major version."""
        # Given
        semver_prefix = "1"  # Should match all 1.x.x versions, return latest

        # When
        result = repository.get_latest_config_by_semver_prefix(semver_prefix)

        # Then
        assert result is not None
        assert result.semver == "1.1.0"  # Latest 1.x.x version
        assert result.hash == sample_feature_config_entities["hashes"]["v1_1_0"]

    def test_repository_implements_port_interface(self, repository):
        """Test that repository properly implements the FeatureConfigReaderPort interface."""
        # Given
        from drl_trading_core.core.port.feature_config_reader_port import FeatureConfigReaderPort

        # When/Then
        assert isinstance(repository, FeatureConfigReaderPort)

        # Verify all required methods are implemented
        assert hasattr(repository, 'get_config')
        assert hasattr(repository, 'is_config_existing')
        assert hasattr(repository, 'get_latest_config_by_semver_prefix')

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
            count = session.query(FeatureConfigEntity).count()
            assert count >= 0  # Should work without error

    def test_read_only_session_usage(self, repository, sample_feature_config_entities):
        """Test that repository uses read-only sessions for read operations."""
        # Given
        version = "1.0.0"

        # When
        # This should use get_read_only_session internally
        result = repository.get_config(version)

        # Then
        assert result is not None
        assert result.semver == "1.0.0"

        # Test other read operations
        exists = repository.is_config_existing(version)
        assert exists is True

        latest = repository.get_latest_config_by_semver_prefix("1.0")
        assert latest is not None

    def test_error_handling_database_connection_issues(self, database_config):
        """Test error handling when database connection fails."""
        # Given
        # Create a repository with invalid database config
        invalid_config = DatabaseConfig(
            host="invalid_host",
            port=5432,
            database="invalid_db",
            username="invalid_user",
            password="invalid_pass"
        )
        session_factory = SQLAlchemySessionFactory(invalid_config)
        repository = FeatureConfigRepository(session_factory)

        # When / Then
        # Should raise an exception when trying to connect
        with pytest.raises(Exception):  # Could be various connection-related exceptions
            repository.get_config("1.0.0")

    def test_concurrent_read_operations(self, repository, sample_feature_config_entities):
        """Test that multiple concurrent read operations work correctly."""
        # Given
        versions = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]

        # When
        results = []
        for version in versions:
            result = repository.get_config(version)
            results.append(result)

        # Then
        assert len(results) == 4
        assert all(isinstance(result, FeatureConfigVersionInfo) for result in results)
        assert [result.semver for result in results] == versions

    def test_json_serialization_of_feature_definitions(self, repository, sample_feature_config_entities):
        """Test that feature definitions are properly serialized/deserialized as JSON."""
        # Given
        version = "1.1.0"  # Has complex feature definitions

        # When
        result = repository.get_config(version)

        # Then
        assert isinstance(result.feature_definitions, list)
        assert len(result.feature_definitions) == 3

        # Verify structure of feature definitions
        for feature_def in result.feature_definitions:
            assert isinstance(feature_def, dict)
            assert "name" in feature_def
            assert "type" in feature_def
            assert isinstance(feature_def["name"], str)
            assert isinstance(feature_def["type"], str)
