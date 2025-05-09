# --- Statistical Significance Validation Config ---
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set, Union


@dataclass(frozen=True)
class StatisticalSignificanceConfig:
    """
    Configuration for statistical significance testing of trading strategies.

    This validator uses formal hypothesis testing to determine if a strategy's
    performance is statistically significant or could be explained by chance.

    Attributes:
        significance_level: Alpha level for hypothesis testing (0-1). Default is 0.05.
        tests: Set of statistical tests to run. Default includes basic t-test and bootstrap.
        correction_method: Method to correct for multiple comparisons. Default is 'holm'.
        null_hypothesis: The null model to test against.
            - 'random_walk': Tests if returns differ from a random walk
            - 'zero_mean': Tests if mean return is different from zero
            - 'benchmark': Tests if returns exceed benchmark returns
        benchmark_symbol: Symbol to use for benchmark comparison (e.g., "SPY").
        min_sample_size: Minimum number of trades/returns required. Default is 30.
        return_type: Type of returns to analyze ('trade' or 'time'). Default is 'time'.
        custom_parameters: Additional parameters for specific tests.
    """

    significance_level: float = 0.05
    tests: Set[str] = field(default_factory=lambda: {"t_test", "bootstrap"})
    correction_method: Literal["bonferroni", "holm", "fdr", "none"] = "holm"
    null_hypothesis: Literal["random_walk", "zero_mean", "benchmark"] = "zero_mean"
    benchmark_symbol: Optional[str] = None
    min_sample_size: int = 30
    return_type: Literal["trade", "time"] = "time"
    custom_parameters: Dict[str, Union[float, int, str]] = field(default_factory=dict)
