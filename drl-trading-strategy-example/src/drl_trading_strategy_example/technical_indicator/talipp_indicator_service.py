import logging
import threading
from typing import Dict, Optional

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry_interface import (
    IndicatorClassRegistryInterface,
)
from injector import inject
from pandas import DataFrame

logger = logging.getLogger(__name__)


@inject
class TaLippIndicatorService(ITechnicalIndicatorFacade):
    """
    Thread-safe technical indicator service for concurrent access.

    This service manages indicator instances with proper thread synchronization
    to handle concurrent access from multiple Dask processes or threads.

    Key thread-safety features:
    - All operations on instances dictionary are protected by locks
    - Atomic check-and-set operations prevent race conditions
    - Defensive copying where appropriate
    - Thread-safe instance management
    """

    def __init__(self, registry: IndicatorClassRegistryInterface) -> None:
        self.instances: Dict[str, BaseIndicator] = {}
        self.registry = registry
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def register_instance(self, name: str, indicator_type: str, **params) -> None:
        """
        Thread-safe registration of indicator instances.

        Args:
            name: Unique name for the indicator instance
            indicator_type: String identifier for indicator type (e.g., "rsi", "ema", "macd")
            **params: Parameters to pass to indicator constructor

        Raises:
            ValueError: If indicator type not found

        Note:
            If an indicator with the same name already exists, this method is idempotent
            and will skip registration to allow feature reuse across warmup and computation phases.
        """
        with self._lock:
            # Idempotent: skip if indicator already exists (allows feature reuse)
            if name in self.instances:
                logger.debug(f"Indicator '{name}' already registered, skipping duplicate registration")
                return

            # Convert string to enum for internal processing
            try:
                enum_type = IndicatorTypeEnum(indicator_type)
            except ValueError:
                raise ValueError(f"Unsupported indicator type: '{indicator_type}'") from None

            # Get indicator class (registry is thread-safe)
            indicator_class = self.registry.get_indicator_class(enum_type)
            if not indicator_class:
                raise ValueError(f"No indicator class found for type {indicator_type}")

            # Atomic assignment
            self.instances[name] = indicator_class(**params)

    def get_all(self, name: str) -> Optional[DataFrame]:
        """
        Thread-safe retrieval of all indicator values.

        Args:
            name: Name of the indicator instance

        Returns:
            DataFrame with all computed values or None if not found

        Raises:
            KeyError: If indicator instance not found
        """
        with self._lock:
            if name not in self.instances:
                raise KeyError(f"Indicator instance '{name}' not found")
            return self.instances[name].get_all()

    def add(self, name: str, values: DataFrame) -> None:
        """
        Thread-safe addition of new values to indicator.

        Args:
            name: Name of the indicator instance
            values: New values to add

        Raises:
            KeyError: If indicator instance not found
        """
        with self._lock:
            if name not in self.instances:
                raise KeyError(f"Indicator instance '{name}' not found")
            self.instances[name].add(values)

    def get_latest(self, name: str) -> Optional[DataFrame]:
        """
        Thread-safe retrieval of latest indicator value.

        Args:
            name: Name of the indicator instance

        Returns:
            DataFrame with latest computed value or None if not found

        Raises:
            KeyError: If indicator instance not found
        """
        with self._lock:
            if name not in self.instances:
                raise KeyError(f"Indicator instance '{name}' not found")
            return self.instances[name].get_latest()

    def is_registered(self, name: str) -> bool:
        """
        Thread-safe check if indicator instance exists.

        Args:
            name: Name to check

        Returns:
            True if instance exists, False otherwise
        """
        with self._lock:
            return name in self.instances

    def unregister_instance(self, name: str) -> bool:
        """
        Thread-safe removal of indicator instance.

        Args:
            name: Name of the indicator instance to remove

        Returns:
            True if instance was found and removed, False otherwise
        """
        with self._lock:
            if name in self.instances:
                del self.instances[name]
                return True
            return False

    def get_registered_names(self) -> list[str]:
        """
        Thread-safe retrieval of all registered indicator names.

        Returns:
            List of registered indicator names (defensive copy)
        """
        with self._lock:
            return list(self.instances.keys())

    def reset(self) -> None:
        """Thread-safe removal of all indicator instances."""
        with self._lock:
            self.instances.clear()

    def __len__(self) -> int:
        """Thread-safe count of registered indicators."""
        with self._lock:
            return len(self.instances)

    def __contains__(self, name: str) -> bool:
        """Thread-safe containment check."""
        return self.is_registered(name)

    def __repr__(self) -> str:
        """Thread-safe string representation."""
        with self._lock:
            return f"TaLippIndicatorService({len(self.instances)} indicators)"
