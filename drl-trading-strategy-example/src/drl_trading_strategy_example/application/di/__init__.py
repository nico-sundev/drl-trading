"""DI modules for strategy example package.

FeatureComputationModule: Features + indicators (all services)
RLEnvironmentModule: Gym environment (training only)
"""

from drl_trading_strategy_example.infrastructure.di.feature_computation_module import (
    FeatureComputationModule,
)
from drl_trading_strategy_example.infrastructure.di.rl_environment_module import (
    RLEnvironmentModule,
)

__all__ = [
    "FeatureComputationModule",
    "RLEnvironmentModule",
]
