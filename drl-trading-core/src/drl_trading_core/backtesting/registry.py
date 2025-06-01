# --- Registry ---
from typing import Any, Dict, Tuple, Type

from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)

VALIDATION_REGISTRY: Dict[str, Tuple[Type[BaseValidationAlgorithm], Type[Any]]] = {}


def register_validation(
    name: str, alg_cls: Type[BaseValidationAlgorithm], config_cls: Type[Any]
) -> None:
    VALIDATION_REGISTRY[name] = (alg_cls, config_cls)
