from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class FeatureDefinition:
    """Feature definition configuration as a frozen dataclass.

    Uses string-based names to avoid circular dependencies between common library
    and strategy-specific enums. The strategy layer can provide type-safe conversion
    to enums when needed.

    Features can be configured with parameter sets (for configurable features like RSI)
    or without any parameters (for simple features like close price).
    """
    name: str  # Feature type identifier (e.g., "rsi", "macd", "close_price")
    enabled: bool
    derivatives: List[int]
    raw_parameter_sets: List[dict] = field(default_factory=list)
