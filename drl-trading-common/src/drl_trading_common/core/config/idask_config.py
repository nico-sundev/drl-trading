from typing import Optional, Protocol


class IDaskConfig(Protocol):
    """Protocol for Dask configuration in domain layer."""

    scheduler: str
    num_workers: Optional[int]
