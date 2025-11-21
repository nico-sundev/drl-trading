"""
Integration tests for FeatureConfig adapters using testcontainers with real PostgreSQL.

This test suite validates the complete feature configuration functionality
against a real PostgreSQL database using testcontainers for isolation.
"""

import pytest
from datetime import datetime
from testcontainers.postgres import PostgresContainer

from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity, Base
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.database.repository.feature_config_repository import FeatureConfigRepository
from drl_trading_training.adapter.feature_config.feature_config_writer import FeatureConfigWriter
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class TestFeatureConfigIntegration:
    """Integration test suite for FeatureConfig adapters with real PostgreSQL."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Start PostgreSQL container for integration tests."""
        with PostgresContainer("postgres:15") as postgres:
            yield postgres

    @pytest.fixture(scope="class")
    def database_config(self, postgres_container):
        """Create database configuration from container."""
        return DatabaseConfig(
            host=postgres_container.get_container_host_ip(),
            port=postgres_container.get_exposed_port(5432),
            database=postgres_container.dbname,
            username=postgres_container.username,
            password=postgres_container.password
        )

    @pytest.fixture(scope="class")
    def session_factory(self, database_config):
        """Create session factory with test database."""
        return SQLAlchemySessionFactory(database_config)

    @pytest.fixture(scope="class", autouse=True)
    def setup_database_schema(self, session_factory):
        """Create database schema for tests."""
        # Create all tables
        Base.metadata.create_all(session_factory._engine)
        yield
        # Cleanup after tests
        Base.metadata.drop_all(session_factory._engine)

    @pytest.fixture
    def reader(self, session_factory):
        """Create reader instance for testing."""
        return FeatureConfigRepository(session_factory)

    @pytest.fixture
    def writer(self, session_factory):
        """Create writer instance for testing."""
        return FeatureConfigWriter(session_factory)

    @pytest.fixture(autouse=True)
    def clean_database(self, session_factory):
        """Clean database before each test."""
        with session_factory.get_session() as session:
            session.query(FeatureConfigEntity).delete()
            session.commit()
        yield

    @pytest.fixture
    def sample_config(self):
        """Sample feature configuration for testing."""
        return FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="abc123def456",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[
                {"name": "rsi_14", "type": "technical_indicator", "period": 14},
                {"name": "sma_50", "type": "moving_average", "period": 50}
            ],
            description="Test feature configuration"
        )

    def test_write_and_read_config_complete_flow(self, writer, reader, sample_config):
        """Test complete write -> read flow with real database."""
        # Given
        config = sample_config

        # When - Write config
        saved_hash = writer.save_config(config)

        # Then - Verify write succeeded
        assert saved_hash == config.hash

        # When - Read config by hash
        retrieved_by_hash = reader.get_config(config.hash)

        # Then - Verify read by hash
        assert retrieved_by_hash.hash == config.hash
        assert retrieved_by_hash.semver == config.semver
        assert retrieved_by_hash.feature_definitions == config.feature_definitions
        assert retrieved_by_hash.description == config.description

        # When - Read config by semver
        retrieved_by_semver = reader.get_config(config.semver)

        # Then - Verify read by semver
        assert retrieved_by_semver.hash == config.hash
        assert retrieved_by_semver.semver == config.semver

    def test_upsert_behavior_complete_flow(self, writer, reader):
        """Test UPSERT behavior - writing same hash updates existing record."""
        # Given
        original_config = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="duplicate_hash",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "original_feature"}],
            description="Original description"
        )

        updated_config = FeatureConfigVersionInfo(
            semver="1.1.0",  # Different semver
            hash="duplicate_hash",  # Same hash
            created_at=datetime(2024, 1, 2, 12, 0, 0),
            feature_definitions=[{"name": "updated_feature"}],
            description="Updated description"
        )

        # When - Save original config
        writer.save_config(original_config)

        # Save updated config with same hash
        writer.save_config(updated_config)

        # Then - Should have updated, not duplicated
        retrieved = reader.get_config("duplicate_hash")
        assert retrieved.semver == "1.1.0"  # Updated semver
        assert retrieved.feature_definitions == [{"name": "updated_feature"}]
        assert retrieved.description == "Updated description"

    def test_config_existence_check(self, writer, reader, sample_config):
        """Test config existence checking."""
        # Given
        config = sample_config

        # When - Check before saving
        exists_before = reader.is_config_existing(config.hash)

        # Then
        assert exists_before is False

        # When - Save config and check again
        writer.save_config(config)
        exists_after = reader.is_config_existing(config.hash)

        # Then
        assert exists_after is True

        # When - Check with semver
        exists_by_semver = reader.is_config_existing(config.semver)

        # Then
        assert exists_by_semver is True

    def test_latest_config_by_semver_prefix(self, writer, reader):
        """Test getting latest config by semver prefix."""
        # Given
        configs = [
            FeatureConfigVersionInfo(
                semver="1.0.0",
                hash="v1_0_0",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[{"name": "v1_0_0_feature"}]
            ),
            FeatureConfigVersionInfo(
                semver="1.0.1",
                hash="v1_0_1",
                created_at=datetime(2024, 1, 2, 12, 0, 0),
                feature_definitions=[{"name": "v1_0_1_feature"}]
            ),
            FeatureConfigVersionInfo(
                semver="1.1.0",
                hash="v1_1_0",
                created_at=datetime(2024, 1, 3, 12, 0, 0),
                feature_definitions=[{"name": "v1_1_0_feature"}]
            )
        ]

        # When - Save all configs
        for config in configs:
            writer.save_config(config)

        # Get latest 1.0.x version
        latest_1_0 = reader.get_latest_config_by_semver_prefix("1.0")

        # Then
        assert latest_1_0 is not None
        assert latest_1_0.semver == "1.0.1"  # Latest 1.0.x
        assert latest_1_0.hash == "v1_0_1"

        # When - Get latest 1.x version
        latest_1 = reader.get_latest_config_by_semver_prefix("1")

        # Then - Should get the most recent overall
        assert latest_1 is not None
        assert latest_1.semver == "1.1.0"

    def test_config_not_found_scenarios(self, reader):
        """Test various config not found scenarios."""
        # When / Then - Nonexistent hash
        with pytest.raises(ValueError, match="Feature configuration version 'nonexistent_hash' not found"):
            reader.get_config("nonexistent_hash")

        # When / Then - Nonexistent semver
        with pytest.raises(ValueError, match="Feature configuration version '99.99.99' not found"):
            reader.get_config("99.99.99")

        # When / Then - Nonexistent prefix
        result = reader.get_latest_config_by_semver_prefix("99.99")
        assert result is None

    def test_complex_feature_definitions_persistence(self, writer, reader):
        """Test saving and retrieving complex feature definitions."""
        # Given
        complex_config = FeatureConfigVersionInfo(
            semver="2.0.0",
            hash="complex_hash",
            created_at=datetime(2024, 2, 1, 12, 0, 0),
            feature_definitions=[
                {
                    "name": "composite_indicator",
                    "type": "custom",
                    "components": [
                        {
                            "name": "rsi",
                            "weight": 0.3,
                            "parameters": {"period": 14, "threshold": [30, 70]}
                        },
                        {
                            "name": "macd",
                            "weight": 0.4,
                            "parameters": {"fast": 12, "slow": 26, "signal": 9}
                        },
                        {
                            "name": "bollinger_bands",
                            "weight": 0.3,
                            "parameters": {"period": 20, "std_dev": 2.0}
                        }
                    ],
                    "aggregation": "weighted_average",
                    "normalization": {
                        "method": "min_max_scaling",
                        "range": [0, 1]
                    }
                }
            ],
            description="Complex multi-component feature configuration"
        )

        # When
        writer.save_config(complex_config)
        retrieved = reader.get_config("complex_hash")

        # Then
        assert retrieved.feature_definitions == complex_config.feature_definitions
        assert retrieved.feature_definitions[0]["components"][0]["parameters"]["threshold"] == [30, 70]
        assert retrieved.feature_definitions[0]["normalization"]["range"] == [0, 1]

    def test_config_with_none_description(self, writer, reader):
        """Test saving and retrieving config with None description."""
        # Given
        config_no_desc = FeatureConfigVersionInfo(
            semver="3.0.0",
            hash="no_desc_hash",
            created_at=datetime(2024, 3, 1, 12, 0, 0),
            feature_definitions=[{"name": "simple_feature"}],
            description=None
        )

        # When
        writer.save_config(config_no_desc)
        retrieved = reader.get_config("no_desc_hash")

        # Then
        assert retrieved.description is None
        assert retrieved.feature_definitions == [{"name": "simple_feature"}]

    def test_multiple_configs_different_hashes(self, writer, reader):
        """Test saving multiple configurations with different hashes."""
        # Given
        configs = [
            FeatureConfigVersionInfo(
                semver="1.0.0",
                hash="hash_a",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[{"name": "feature_a"}],
                description="Config A"
            ),
            FeatureConfigVersionInfo(
                semver="1.0.0",  # Same semver, different hash
                hash="hash_b",
                created_at=datetime(2024, 1, 2, 12, 0, 0),
                feature_definitions=[{"name": "feature_b"}],
                description="Config B"
            )
        ]

        # When
        for config in configs:
            writer.save_config(config)

        # Then - Both should exist
        config_a = reader.get_config("hash_a")
        assert config_a.description == "Config A"
        assert config_a.feature_definitions == [{"name": "feature_a"}]

        config_b = reader.get_config("hash_b")
        assert config_b.description == "Config B"
        assert config_b.feature_definitions == [{"name": "feature_b"}]

        # Both should be found by existence check
        assert reader.is_config_existing("hash_a") is True
        assert reader.is_config_existing("hash_b") is True
