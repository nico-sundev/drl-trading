"""Configuration for feature computation operations.

This module provides service-specific configuration wrappers that encapsulate
the technical details (like Dask parallelization) for specific use cases.
"""

from dataclasses import dataclass

from drl_trading_common.config.dask_config import DaskConfig


@dataclass(frozen=True)
class FeatureComputationConfig:
    """Configuration specifically for feature computation operations.

    This wrapper type allows different services to provide their own
    Dask configuration tuned for feature computation workloads without
    requiring the core FeatureManager to know about service-specific
    configuration collections (like DaskConfigs).

    Attributes:
        dask: Dask parallelization configuration for CPU-bound feature computation.
              Typically uses process-based scheduler for true parallelism with
              hundreds of features across multiple timeframes.

    Example:
        >>> from drl_trading_common.config.dask_config import DaskConfig
        >>> config = FeatureComputationConfig(
        ...     dask=DaskConfig(
        ...         scheduler="processes",
        ...         num_workers=4,
        ...         memory_limit_per_worker_mb=1024
        ...     )
        ... )
    """

    dask: DaskConfig
