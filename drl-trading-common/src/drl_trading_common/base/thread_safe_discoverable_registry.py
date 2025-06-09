"""Thread-safe implementation of DiscoverableRegistry for concurrent access."""

import threading
from typing import Dict, Optional, Type, TypeVar, Generic

from drl_trading_common.base.discoverable_registry import DiscoverableRegistry

# Generic type variables for the base class types
T = TypeVar('T')  # The type of class being registered
K = TypeVar('K')  # The type of key being used


class ThreadSafeDiscoverableRegistry(DiscoverableRegistry[K, T], Generic[K, T]):
    """
    Thread-safe implementation of DiscoverableRegistry.

    This class extends DiscoverableRegistry with thread-safety guarantees for
    concurrent access patterns common in multiprocessing environments like Dask.

    Key thread-safety features:
    - All operations on _class_map are protected by locks
    - Atomic check-and-set operations prevent race conditions
    - Discovery operations are thread-safe
    - Reset operations are properly synchronized

    Type Parameters:
        T: The type of class being registered (e.g., BaseFeature, BaseIndicator)
        K: The type of key being used (e.g., str, Enum)
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def get_class(self, key: K) -> Optional[Type[T]]:
        """
        Thread-safe get operation.

        Args:
            key: The key to look up

        Returns:
            The class if found, None otherwise
        """
        with self._lock:
            return super().get_class(key)

    def register_class(self, key: K, class_type: Type[T]) -> None:
        """
        Thread-safe register operation with atomic check-and-set.

        Args:
            key: The key to register
            class_type: The class to register

        Raises:
            TypeError: If class validation fails
        """
        with self._lock:
            # Validation happens inside the lock to ensure consistency
            self._validate_class(class_type)

            # Atomic check for existing registration
            if key in self._class_map:
                self._handle_duplicate_registration(key, class_type)

            # Atomic assignment
            self._class_map[key] = class_type
            self._log_registration(key, class_type)

    def discover_classes(self, package_name: str) -> Dict[K, Type[T]]:
        """
        Thread-safe discovery operation.

        Args:
            package_name: The name of the package to discover classes from

        Returns:
            A dictionary mapping keys to their corresponding class types
        """
        with self._lock:
            return super().discover_classes(package_name)

    def reset(self) -> None:
        """Thread-safe reset operation."""
        with self._lock:
            super().reset()

    def get_all_registered(self) -> Dict[K, Type[T]]:
        """
        Thread-safe method to get a copy of all registered classes.

        Returns:
            A copy of the class map (defensive copy to prevent external modification)
        """
        with self._lock:
            return self._class_map.copy()

    def is_registered(self, key: K) -> bool:
        """
        Thread-safe check if a key is registered.

        Args:
            key: The key to check

        Returns:
            True if the key is registered, False otherwise
        """
        with self._lock:
            return key in self._class_map

    def unregister_class(self, key: K) -> bool:
        """
        Thread-safe method to unregister a class.

        Args:
            key: The key to unregister

        Returns:
            True if the key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._class_map:
                del self._class_map[key]
                return True
            return False

    def _log_registration(self, key: K, class_type: Type[T]) -> None:
        """Helper method to log registration (already inside lock)."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Registered class '{key}': {class_type}")

    def __len__(self) -> int:
        """Thread-safe method to get the number of registered classes."""
        with self._lock:
            return len(self._class_map)

    def __contains__(self, key: K) -> bool:
        """Thread-safe containment check."""
        return self.is_registered(key)

    def __repr__(self) -> str:
        """Thread-safe string representation."""
        with self._lock:
            return f"{self.__class__.__name__}({len(self._class_map)} registered classes)"
