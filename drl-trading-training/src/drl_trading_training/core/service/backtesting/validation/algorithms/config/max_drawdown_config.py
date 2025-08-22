# --- Max Drawdown Validation ---
from dataclasses import dataclass


@dataclass(frozen=True)
class MaxDrawdownConfig:
    max_drawdown_pct: float = 10.0  # maximum acceptable drawdown in percent
