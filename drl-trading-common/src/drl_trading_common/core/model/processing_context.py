"""
Processing context enumeration for feature preprocessing workflows.

This module defines the different processing contexts that determine
how feature preprocessing requests are handled throughout the system.
"""

from enum import Enum


class ProcessingContext(str, Enum):
    """
    Processing context for feature preprocessing requests.

    Defines the operational context that determines how preprocessing
    requests are handled. Configuration and behavior rules are defined
    externally via service configuration, not in this enum.

    Attributes:
        TRAINING: Offline training data preparation
        INFERENCE: Real-time inference feature computation
        BACKFILL: Historical data gap filling
        CATCHUP: Initial catchup before live streaming
    """

    TRAINING = "training"
    INFERENCE = "inference"
    BACKFILL = "backfill"
    CATCHUP = "catchup"
