"""Feature store adapter implementations (Feast + offline repositories).

Currently includes Feast provider and wrapper migrated from core. Additional offline
repository implementations will be migrated in subsequent iterations.
"""
from .feature_store_fetch_repository import FeatureStoreFetchRepository
from .util.feature_store_utilities import get_feature_service_name, get_feature_view_name

__all__ = [
    "FeatureStoreFetchRepository",
    "get_feature_service_name",
    "get_feature_view_name"
]
