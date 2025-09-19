"""
Unit tests for FeatureConfigMapper.

Tests the mapper functionality for converting between FeatureConfigVersionInfo
domain models and FeatureConfigEntity database entities.
"""

import pytest
from datetime import datetime

from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity
from drl_trading_adapter.adapter.database.mapper.feature_config_mapper import FeatureConfigMapper
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo


class TestFeatureConfigMapper:
    """Unit test suite for FeatureConfigMapper."""

    @pytest.fixture
    def sample_domain_model(self) -> FeatureConfigVersionInfo:
        """Sample domain model for testing."""
        return FeatureConfigVersionInfo(
            semver="1.2.3",
            hash="abc123def456ghi789",
            created_at=datetime(2024, 1, 15, 14, 30, 0),
            feature_definitions=[
                {"name": "rsi_14", "type": "technical_indicator", "period": 14},
                {"name": "sma_50", "type": "moving_average", "period": 50}
            ],
            description="Test feature configuration"
        )

    @pytest.fixture
    def sample_entity(self) -> FeatureConfigEntity:
        """Sample entity for testing."""
        return FeatureConfigEntity(
            hash="xyz789uvw456rst123",
            semver="2.1.0",
            created_at=datetime(2024, 2, 20, 10, 15, 0),
            feature_definitions=[
                {"name": "macd", "type": "oscillator", "fast": 12, "slow": 26, "signal": 9},
                {"name": "bollinger_bands", "type": "volatility", "period": 20, "std_dev": 2}
            ],
            description="Entity test configuration"
        )

    def test_to_entity_conversion(self, sample_domain_model):
        """Test conversion from domain model to entity."""
        # Given
        domain_model = sample_domain_model

        # When
        entity = FeatureConfigMapper.to_entity(domain_model)

        # Then
        assert isinstance(entity, FeatureConfigEntity)
        assert entity.hash == domain_model.hash
        assert entity.semver == domain_model.semver
        assert entity.created_at == domain_model.created_at
        assert entity.feature_definitions == domain_model.feature_definitions
        assert entity.description == domain_model.description

    def test_to_domain_model_conversion(self, sample_entity):
        """Test conversion from entity to domain model."""
        # Given
        entity = sample_entity

        # When
        domain_model = FeatureConfigMapper.to_domain_model(entity)

        # Then
        assert isinstance(domain_model, FeatureConfigVersionInfo)
        assert domain_model.hash == entity.hash
        assert domain_model.semver == entity.semver
        assert domain_model.created_at == entity.created_at
        assert domain_model.feature_definitions == entity.feature_definitions
        assert domain_model.description == entity.description

    def test_round_trip_conversion(self, sample_domain_model):
        """Test domain model -> entity -> domain model conversion."""
        # Given
        original_model = sample_domain_model

        # When
        entity = FeatureConfigMapper.to_entity(original_model)
        converted_model = FeatureConfigMapper.to_domain_model(entity)

        # Then
        assert converted_model.hash == original_model.hash
        assert converted_model.semver == original_model.semver
        assert converted_model.created_at == original_model.created_at
        assert converted_model.feature_definitions == original_model.feature_definitions
        assert converted_model.description == original_model.description

    def test_to_entity_with_none_description(self):
        """Test conversion to entity with None description."""
        # Given
        domain_model = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}],
            description=None
        )

        # When
        entity = FeatureConfigMapper.to_entity(domain_model)

        # Then
        assert entity.description is None

    def test_to_entity_validation_missing_semver(self):
        """Test validation when semver is missing."""
        # Given
        domain_model = FeatureConfigVersionInfo(
            semver="",  # Empty semver
            hash="test_hash",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        with pytest.raises(ValueError, match="semver is required for entity mapping"):
            FeatureConfigMapper.to_entity(domain_model)

    def test_to_entity_validation_missing_hash(self):
        """Test validation when hash is missing."""
        # Given
        domain_model = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="",  # Empty hash
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        with pytest.raises(ValueError, match="hash is required for entity mapping"):
            FeatureConfigMapper.to_entity(domain_model)

    def test_to_entity_validation_missing_created_at(self):
        """Test validation when created_at is missing."""
        # Given
        domain_model = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=None,  # None created_at
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        with pytest.raises(ValueError, match="created_at is required for entity mapping"):
            FeatureConfigMapper.to_entity(domain_model)

    def test_to_entity_validation_missing_feature_definitions(self):
        """Test validation when feature_definitions is missing."""
        # Given
        domain_model = FeatureConfigVersionInfo(
            semver="1.0.0",
            hash="test_hash",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[]  # Empty list
        )

        # When / Then
        with pytest.raises(ValueError, match="feature_definitions is required for entity mapping"):
            FeatureConfigMapper.to_entity(domain_model)

    def test_to_domain_model_validation_missing_hash(self):
        """Test validation when entity hash is missing."""
        # Given
        entity = FeatureConfigEntity(
            hash="",  # Empty hash
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        with pytest.raises(ValueError, match="Entity hash cannot be None or empty"):
            FeatureConfigMapper.to_domain_model(entity)

    def test_to_domain_model_validation_missing_semver(self):
        """Test validation when entity semver is missing."""
        # Given
        entity = FeatureConfigEntity(
            hash="test_hash",
            semver="",  # Empty semver
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=[{"name": "test"}]
        )

        # When / Then
        with pytest.raises(ValueError, match="Entity semver cannot be None or empty"):
            FeatureConfigMapper.to_domain_model(entity)

    def test_to_domain_model_validation_none_feature_definitions(self):
        """Test validation when entity feature_definitions is None."""
        # Given
        entity = FeatureConfigEntity(
            hash="test_hash",
            semver="1.0.0",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            feature_definitions=None  # None feature_definitions
        )

        # When / Then
        with pytest.raises(ValueError, match="Entity feature_definitions cannot be None"):
            FeatureConfigMapper.to_domain_model(entity)

    def test_to_domain_models_list_conversion(self):
        """Test converting list of entities to list of domain models."""
        # Given
        entities = [
            FeatureConfigEntity(
                hash="hash1",
                semver="1.0.0",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[{"name": "feature1"}],
                description="First config"
            ),
            FeatureConfigEntity(
                hash="hash2",
                semver="2.0.0",
                created_at=datetime(2024, 2, 1, 12, 0, 0),
                feature_definitions=[{"name": "feature2"}],
                description="Second config"
            )
        ]

        # When
        domain_models = FeatureConfigMapper.to_domain_models(entities)

        # Then
        assert len(domain_models) == 2
        assert all(isinstance(model, FeatureConfigVersionInfo) for model in domain_models)
        assert domain_models[0].hash == "hash1"
        assert domain_models[0].semver == "1.0.0"
        assert domain_models[1].hash == "hash2"
        assert domain_models[1].semver == "2.0.0"

    def test_to_entities_list_conversion(self):
        """Test converting list of domain models to list of entities."""
        # Given
        domain_models = [
            FeatureConfigVersionInfo(
                hash="hash1",
                semver="1.0.0",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                feature_definitions=[{"name": "feature1"}],
                description="First config"
            ),
            FeatureConfigVersionInfo(
                hash="hash2",
                semver="2.0.0",
                created_at=datetime(2024, 2, 1, 12, 0, 0),
                feature_definitions=[{"name": "feature2"}],
                description="Second config"
            )
        ]

        # When
        entities = FeatureConfigMapper.to_entities(domain_models)

        # Then
        assert len(entities) == 2
        assert all(isinstance(entity, FeatureConfigEntity) for entity in entities)
        assert entities[0].hash == "hash1"
        assert entities[0].semver == "1.0.0"
        assert entities[1].hash == "hash2"
        assert entities[1].semver == "2.0.0"

    def test_empty_list_conversions(self):
        """Test converting empty lists."""
        # Given / When / Then
        assert FeatureConfigMapper.to_domain_models([]) == []
        assert FeatureConfigMapper.to_entities([]) == []

    def test_complex_feature_definitions_conversion(self):
        """Test conversion with complex nested feature definitions."""
        # Given
        complex_features = [
            {
                "name": "composite_indicator",
                "type": "custom",
                "components": [
                    {"name": "rsi", "weight": 0.3, "params": {"period": 14}},
                    {"name": "macd", "weight": 0.4, "params": {"fast": 12, "slow": 26}},
                    {"name": "bollinger", "weight": 0.3, "params": {"period": 20, "std": 2}}
                ],
                "aggregation": "weighted_average",
                "normalization": "min_max_scaling"
            }
        ]

        domain_model = FeatureConfigVersionInfo(
            hash="complex_hash",
            semver="3.0.0",
            created_at=datetime(2024, 3, 1, 12, 0, 0),
            feature_definitions=complex_features,
            description="Complex feature configuration"
        )

        # When
        entity = FeatureConfigMapper.to_entity(domain_model)
        converted_model = FeatureConfigMapper.to_domain_model(entity)

        # Then
        assert converted_model.feature_definitions == complex_features
        assert converted_model.feature_definitions[0]["components"][0]["weight"] == 0.3
        assert converted_model.feature_definitions[0]["aggregation"] == "weighted_average"
