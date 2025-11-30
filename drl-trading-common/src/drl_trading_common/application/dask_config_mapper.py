from drl_trading_common.config.dask_config import DaskConfig
from drl_trading_common.core.config.domain_dask_config import DomainDaskConfig


def map_to_domain_dask_config(dask_config: DaskConfig) -> DomainDaskConfig:
    """Map Pydantic DaskConfig to domain DomainDaskConfig.

    Args:
        dask_config: Pydantic DaskConfig instance

    Returns:
        DomainDaskConfig instance
    """
    return DomainDaskConfig(
        scheduler=dask_config.scheduler,
        num_workers=dask_config.num_workers,
    )
