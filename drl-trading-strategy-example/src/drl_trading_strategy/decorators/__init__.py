"""
Decorators for DRL trading strategy components.
"""

from .feature_type_decorator import feature_type, get_feature_type_from_class

__all__ = [
    "feature_type",
    "get_feature_type_from_class",
]
