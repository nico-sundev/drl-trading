from dataclasses import dataclass, field
from typing import Any, Dict, List
from .base_parameter_set_config import BaseParameterSetConfig


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
    parameter_sets: List[Dict[str, Any]] = field(default_factory=list)  # raw input from JSON, can be empty
    parsed_parameter_sets: Dict[str, BaseParameterSetConfig] = field(default_factory=dict)
