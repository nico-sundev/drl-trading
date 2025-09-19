"""
SQLAlchemy entity for feature_configs table.

This module defines the FeatureConfigEntity class that maps to the feature_configs
database table, providing ORM functionality for feature configuration storage.
"""

from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.sql import func

from drl_trading_adapter.adapter.database.entity.market_data_entity import Base


class FeatureConfigEntity(Base):
    """
    SQLAlchemy entity for feature_configs table.

    Maps to the feature_configs table which stores versioned feature engineering
    configurations for reproducible model training and inference.

    Table structure:
    - hash: Primary key, unique identifier for the configuration
    - semver: Semantic version string for human-readable versioning
    - created_at: Timestamp when the configuration was created
    - feature_definitions: JSON blob containing the feature configuration
    - description: Optional human-readable description
    """

    __tablename__ = "feature_configs"

    # Primary key - unique hash of the configuration
    hash = Column(String, primary_key=True, nullable=False)

    # Semantic version for human-readable versioning
    semver = Column(String, nullable=False)

    # Timestamp when configuration was created
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # JSON blob containing feature definitions
    feature_definitions = Column(JSON, nullable=False)

    # Optional description of the configuration
    description = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of the entity."""
        return (
            f"FeatureConfigEntity(hash='{self.hash}', "
            f"semver='{self.semver}', "
            f"created_at='{self.created_at}', "
            f"description='{self.description}')"
        )

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on hash (primary key)."""
        if not isinstance(other, FeatureConfigEntity):
            return False
        return self.hash == other.hash

    def __hash__(self) -> int:
        """Hash function based on the configuration hash."""
        return hash(self.hash) if self.hash else 0
