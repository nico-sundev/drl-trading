from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    score: Optional[float]
    threshold: Optional[float | tuple]
    meta: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
