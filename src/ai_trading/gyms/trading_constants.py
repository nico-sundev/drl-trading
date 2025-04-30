"""Constants used throughout the trading environment components."""

from typing import List

# Primary context columns that must be present in raw data
PRIMARY_CONTEXT_COLUMNS: List[str] = ["Time", "High", "Low", "Close"]

# Additional primary context columns that might be used if available
OPTIONAL_PRIMARY_COLUMNS: List[str] = ["Open", "Volume"]

# All primary context columns combined (those that should exist in raw data)
ALL_PRIMARY_COLUMNS: List[str] = PRIMARY_CONTEXT_COLUMNS + OPTIONAL_PRIMARY_COLUMNS

# Derived context columns that are computed from primary columns
DERIVED_CONTEXT_COLUMNS: List[str] = ["Atr"]

# Required context columns for the trading environment (both primary and derived)
REQUIRED_CONTEXT_COLUMNS: List[str] = PRIMARY_CONTEXT_COLUMNS + ["Atr"]

# All context columns combined (both primary and derived, both required and optional)
ALL_CONTEXT_COLUMNS: List[str] = ALL_PRIMARY_COLUMNS + DERIVED_CONTEXT_COLUMNS
