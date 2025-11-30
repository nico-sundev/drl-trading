from dataclasses import dataclass
from typing import Optional

from drl_trading_common.core.config.idask_config import IDaskConfig


@dataclass(frozen=True)
class DomainDaskConfig(IDaskConfig):
    """Domain representation of Dask configuration."""

    scheduler: str
    num_workers: Optional[int]
