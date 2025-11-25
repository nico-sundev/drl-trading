"""
Mapper for converting between FeatureConfigVersionInfo domain models and FeatureConfigEntity.

This module provides bidirectional mapping functionality between the domain model
(FeatureConfigVersionInfo) used in business logic and the database entity
(FeatureConfigEntity) used by SQLAlchemy ORM.
"""

from typing import List

from drl_trading_adapter.adapter.database.entity.feature_config_entity import FeatureConfigEntity
from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo


class FeatureConfigMapper:
    """
    Mapper for converting between domain models and database entities.

    Handles bidirectional conversion between FeatureConfigVersionInfo (domain)
    and FeatureConfigEntity (database) with proper data transformation.
    """

    @staticmethod
    def to_entity(domain_model: FeatureConfigVersionInfo) -> FeatureConfigEntity:
        """
        Convert domain model to database entity.

        Args:
            domain_model: FeatureConfigVersionInfo domain object

        Returns:
            FeatureConfigEntity: Corresponding database entity

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not domain_model.semver:
            raise ValueError("semver is required for entity mapping")
        if not domain_model.hash:
            raise ValueError("hash is required for entity mapping")
        if not domain_model.created_at:
            raise ValueError("created_at is required for entity mapping")
        if not domain_model.feature_definitions:
            raise ValueError("feature_definitions is required for entity mapping")

        return FeatureConfigEntity(
            hash=domain_model.hash,
            semver=domain_model.semver,
            created_at=domain_model.created_at,
            feature_definitions=domain_model.feature_definitions,
            description=domain_model.description
        )

    @staticmethod
    def to_domain_model(entity: FeatureConfigEntity) -> FeatureConfigVersionInfo:
        """
        Convert database entity to domain model.

        Args:
            entity: FeatureConfigEntity database object

        Returns:
            FeatureConfigVersionInfo: Corresponding domain model

        Raises:
            ValueError: If entity data is invalid or incomplete
        """
        if not entity.hash:
            raise ValueError("Entity hash cannot be None or empty")
        if not entity.semver:
            raise ValueError("Entity semver cannot be None or empty")
        if not entity.created_at:
            raise ValueError("Entity created_at cannot be None")
        if entity.feature_definitions is None:
            raise ValueError("Entity feature_definitions cannot be None")

        return FeatureConfigVersionInfo(
            hash=entity.hash,
            semver=entity.semver,
            created_at=entity.created_at,
            feature_definitions=entity.feature_definitions,
            description=entity.description
        )

    @staticmethod
    def to_domain_models(entities: List[FeatureConfigEntity]) -> List[FeatureConfigVersionInfo]:
        """
        Convert list of database entities to list of domain models.

        Args:
            entities: List of FeatureConfigEntity database objects

        Returns:
            List[FeatureConfigVersionInfo]: List of corresponding domain models
        """
        return [FeatureConfigMapper.to_domain_model(entity) for entity in entities]

    @staticmethod
    def to_entities(domain_models: List[FeatureConfigVersionInfo]) -> List[FeatureConfigEntity]:
        """
        Convert list of domain models to list of database entities.

        Args:
            domain_models: List of FeatureConfigVersionInfo domain objects

        Returns:
            List[FeatureConfigEntity]: List of corresponding database entities
        """
        return [FeatureConfigMapper.to_entity(model) for model in domain_models]
