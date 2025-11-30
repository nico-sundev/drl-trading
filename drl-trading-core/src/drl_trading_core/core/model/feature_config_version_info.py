from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from drl_trading_core.core.model.feature_definition import FeatureDefinition


@dataclass
class FeatureConfigVersionInfo:
    semver: str
    hash: str
    created_at: datetime
    feature_definitions: List[FeatureDefinition]
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "semver": self.semver,
            "hash": self.hash,
            "created_at": self.created_at.isoformat(),
            "feature_definitions": [fd.__dict__ for fd in self.feature_definitions],
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: dict) -> "FeatureConfigVersionInfo":
        return FeatureConfigVersionInfo(
            semver=data["semver"],
            hash=data["hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            feature_definitions=[FeatureDefinition(**fd) for fd in data["feature_definitions"]],
            description=data.get("description"),
        )
