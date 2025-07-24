"""
Enum for offline feature repository strategy types.

This enum defines the available storage backends for offline feature repositories.
"""

from enum import Enum


class OfflineRepoStrategyEnum(Enum):
    """
    Enumeration of available offline feature repository strategies.

    Used to configure which storage backend should be used for offline feature storage.
    """
    LOCAL = "local"
    S3 = "s3"
