from drl_trading_common.application.dask_config_mapper import map_to_domain_dask_config
from drl_trading_preprocess.core.config.feature_coverage_analyzer_config import (
    FeatureCoverageAnalyzerConfig,
)
from drl_trading_preprocess.core.config.preprocessing_orchestrator_config import (
    PreprocessingOrchestratorConfig,
)
from drl_trading_preprocess.infrastructure.config.preprocess_config import (
    PreprocessConfig,
)


def create_preprocessing_orchestrator_config(
    config: PreprocessConfig,
) -> PreprocessingOrchestratorConfig:
    """Create PreprocessingOrchestratorConfig from PreprocessConfig.

    Args:
        config: PreprocessConfig instance
    """
    return PreprocessingOrchestratorConfig(
        num_warmup_candles=config.feature_computation_config.warmup_candles,
        feature_coverage_analyzer_config=FeatureCoverageAnalyzerConfig(
            dask_config=map_to_domain_dask_config(config.dask_configs.coverage_analysis),
        ),
    )
