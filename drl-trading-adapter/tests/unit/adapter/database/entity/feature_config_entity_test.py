"""
Unit tests for FeatureConfigEntity.

Tests the SQLAlchemy entity for feature configuration storage,
validating column mappings, constraints, and ORM functionality.
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity, Base


class TestFeatureConfigEntity:
    """Unit test suite for FeatureConfigEntity."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create in-memory SQLite engine for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create database session for each test."""
        Session = sessionmaker(bind=engine)
        session = Session()

        # Clean up any existing data before each test
        session.query(FeatureConfigEntity).delete()
        session.commit()

        yield session
        session.rollback()
        session.close()

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
            description="Initial feature configuration"
        )

    def test_entity_creation(self, sample_entity):
        """Test basic entity creation and attribute access."""
        # Given / When
        entity = sample_entity

        # Then
        assert entity.hash == "abc123def456"
        assert entity.semver == "1.0.0"
        assert entity.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert len(entity.feature_definitions) == 2
        assert entity.description == "Initial feature configuration"

    def test_entity_creation_without_description(self):
        """Test entity creation with optional description field."""
        # Given / When
        entity = FeatureConfigEntity(
            hash="def789ghi012",
            semver="2.0.0",
            created_at=datetime(2024, 2, 1, 12, 0, 0),
            feature_definitions=[{"name": "macd", "type": "oscillator"}]
        )

        # Then
        assert entity.hash == "def789ghi012"
        assert entity.semver == "2.0.0"
        assert entity.description is None
        assert len(entity.feature_definitions) == 1

    def test_entity_save_and_retrieve(self, session, sample_entity):
        """Test saving and retrieving entity from database."""
        # Given
        entity = sample_entity

        # When
        session.add(entity)
        session.commit()

        # Then
        retrieved = session.query(FeatureConfigEntity).filter_by(hash="abc123def456").first()
        assert retrieved is not None
        assert retrieved.hash == "abc123def456"
        assert retrieved.semver == "1.0.0"
        assert retrieved.feature_definitions == [
            {"name": "rsi_14", "type": "technical_indicator"},
            {"name": "sma_20", "type": "moving_average"}
        ]

    def test_entity_primary_key_constraint(self, session):
        """Test primary key constraint on hash field."""
        # Given
        entity1 = FeatureConfigEntity(
            hash="duplicate_hash",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )
        entity2 = FeatureConfigEntity(
            hash="duplicate_hash",  # Same hash
            semver="2.0.0",
            created_at=datetime(2024, 1, 2, 12, 0, 0),
            feature_definitions=[{"name": "test2"}]
        )

        # When
        session.add(entity1)
        session.commit()

        session.add(entity2)

        # Then
        with pytest.raises(Exception):  # SQLite raises IntegrityError for constraint violation
            session.commit()

    def test_entity_json_field_handling(self, session):
        """Test JSON field storage and retrieval."""
        # Given
        complex_features = [
            {
                "name": "bollinger_bands",
                "type": "volatility_indicator",
                "parameters": {
                    "period": 20,
                    "std_dev": 2.0,
                    "ma_type": "simple"
                },
                "outputs": ["upper_band", "middle_band", "lower_band"]
            },
            {
                "name": "stochastic_oscillator",
                "type": "momentum_indicator",
                "parameters": {
                    "k_period": 14,
                    "d_period": 3,
                    "smooth_k": 1
                }
            }
        ]

        entity = FeatureConfigEntity(
            hash="complex_json_test",
            semver="1.5.0",
            created_at=datetime(2024, 3, 1, 12, 0, 0),
            feature_definitions=complex_features
        )

        # When
        session.add(entity)
        session.commit()

        # Then
        retrieved = session.query(FeatureConfigEntity).filter_by(hash="complex_json_test").first()
        assert retrieved.feature_definitions == complex_features
        assert retrieved.feature_definitions[0]["parameters"]["period"] == 20
        assert retrieved.feature_definitions[1]["type"] == "momentum_indicator"

    def test_entity_equality(self):
        """Test entity equality comparison."""
        # Given
        entity1 = FeatureConfigEntity(
            hash="same_hash",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )
        entity2 = FeatureConfigEntity(
            hash="same_hash",  # Same hash
            semver="2.0.0",    # Different version
            created_at=datetime(2024, 2, 1, 12, 0, 0),
            feature_definitions=[{"name": "different"}]
        )
        entity3 = FeatureConfigEntity(
            hash="different_hash",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        assert entity1 == entity2  # Same hash, should be equal
        assert entity1 != entity3  # Different hash, should not be equal
        assert entity1 != "not_an_entity"  # Different type, should not be equal

    def test_entity_hash_function(self):
        """Test entity hash function."""
        # Given
        entity1 = FeatureConfigEntity(hash="test_hash_123")
        entity2 = FeatureConfigEntity(hash="test_hash_123")
        entity3 = FeatureConfigEntity(hash="different_hash")

        # When / Then
        assert hash(entity1) == hash(entity2)  # Same hash value
        assert hash(entity1) != hash(entity3)  # Different hash value

    def test_entity_repr(self, sample_entity):
        """Test string representation of entity."""
        # Given / When
        repr_str = repr(sample_entity)

        # Then
        assert "FeatureConfigEntity" in repr_str
        assert "hash='abc123def456'" in repr_str
        assert "semver='1.0.0'" in repr_str
        assert "description='Initial feature configuration'" in repr_str

    def test_entity_with_empty_feature_definitions(self, session):
        """Test entity with empty feature definitions list."""
        # Given
        entity = FeatureConfigEntity(
            hash="empty_features",
            semver="0.1.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[]  # Empty list
        )

        # When
        session.add(entity)
        session.commit()

        # Then
        retrieved = session.query(FeatureConfigEntity).filter_by(hash="empty_features").first()
        assert retrieved.feature_definitions == []

    def test_entity_query_by_semver(self, session):
        """Test querying entities by semantic version."""
        # Given
        entities = [
            FeatureConfigEntity(
                hash="v1_config",
                semver="1.0.0",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[{"name": "v1_feature"}]
            ),
            FeatureConfigEntity(
                hash="v2_config",
                semver="2.0.0",
                created_at=datetime(2024, 2, 1, 12, 0, 0),
                feature_definitions=[{"name": "v2_feature"}]
            )
        ]

        # When
        for entity in entities:
            session.add(entity)
        session.commit()

        # Then
        v1_entity = session.query(FeatureConfigEntity).filter_by(semver="1.0.0").first()
        assert v1_entity.hash == "v1_config"
        assert v1_entity.feature_definitions[0]["name"] == "v1_feature"

        v2_entity = session.query(FeatureConfigEntity).filter_by(semver="2.0.0").first()
        assert v2_entity.hash == "v2_config"
