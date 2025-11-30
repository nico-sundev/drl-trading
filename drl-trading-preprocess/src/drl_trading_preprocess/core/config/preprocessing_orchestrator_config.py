from typing import Optional
from attr import dataclass

from drl_trading_preprocess.core.config.feature_coverage_analyzer_config import FeatureCoverageAnalyzerConfig


@dataclass(frozen=True)
class PreprocessingOrchestratorConfig:
    feature_coverage_analyzer_config: FeatureCoverageAnalyzerConfig
    num_warmup_candles: Optional[int] = None
