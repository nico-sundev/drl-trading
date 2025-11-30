from dataclasses import dataclass

from drl_trading_common.core.config.idask_config import IDaskConfig


@dataclass(frozen=True)
class FeatureCoverageAnalyzerConfig:
    dask_config: IDaskConfig
