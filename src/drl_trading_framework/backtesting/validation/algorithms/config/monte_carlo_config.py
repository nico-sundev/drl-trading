# --- Monte Carlo Validation Config ---
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation validation of trading strategies.

    Monte Carlo methods simulate thousands of alternate performance paths to
    determine the robustness of a strategy. This helps assess whether the observed
    performance could be due to luck or represents skill.

    Attributes:
        num_simulations: Number of Monte Carlo simulations to run. Default is 1000.
        confidence_level: Confidence level for statistical tests (0-100). Default is 95.
        simulation_method: Method to generate alternate performance paths.
            - 'returns_bootstrap': Randomly resample returns preserving their distribution
            - 'block_bootstrap': Resample blocks of returns to preserve autocorrelation
            - 'parametric': Generate synthetic returns using a parametric model
        block_length: Length of blocks when using block_bootstrap method. Default is 10.
        min_acceptable_win_rate: Minimum acceptable win rate across simulations. Default is 50%.
        custom_metrics: Additional custom metrics to calculate in simulations.
        random_seed: Optional seed for reproducibility.
    """

    num_simulations: int = 1000
    confidence_level: float = 95.0
    simulation_method: Literal["returns_bootstrap", "block_bootstrap", "parametric"] = (
        "block_bootstrap"
    )
    block_length: int = 10
    min_acceptable_win_rate: float = 50.0
    custom_metrics: Dict[str, Union[float, int]] = field(default_factory=dict)
    random_seed: Optional[int] = None
