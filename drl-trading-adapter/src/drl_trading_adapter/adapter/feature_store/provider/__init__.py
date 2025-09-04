"""Feature store adapter implementations (Feast + offline repositories).

Currently includes Feast provider and wrapper migrated from core. Additional offline
repository implementations will be migrated in subsequent iterations.
"""
from .feast_provider import FeastProvider
from .feature_store_wrapper import FeatureStoreWrapper

__all__ = [
    "FeastProvider",
    "FeatureStoreWrapper",
    "FeatureStoreFetchRepository"
]
