"""Feature store adapter implementations (Feast + offline repositories).

Currently includes Feast provider and wrapper migrated from core. Additional offline
repository implementations will be migrated in subsequent iterations.
"""
from .feast.feast_provider import FeastProvider  # noqa: F401
from .feast.feature_store_wrapper import FeatureStoreWrapper  # noqa: F401

__all__ = [
    "FeastProvider",
    "FeatureStoreWrapper",
]
