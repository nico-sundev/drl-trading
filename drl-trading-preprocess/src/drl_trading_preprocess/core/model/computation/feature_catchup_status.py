from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class FeatureCatchupStatus:
    """Container for detailed feature catch-up status information."""

    all_caught_up: bool
    caught_up_features: List[str]
    not_caught_up_features: List[str]
    total_features: int
    catch_up_percentage: float
    reference_time: datetime

    def has_features_needing_warmup(self) -> bool:
        """Check if any features need warmup."""
        return len(self.not_caught_up_features) > 0

    def get_summary_message(self) -> str:
        """Get human-readable summary."""
        return (
            f"Feature catch-up status: {len(self.caught_up_features)}/{self.total_features} "
            f"({self.catch_up_percentage:.1f}%) caught up"
        )
