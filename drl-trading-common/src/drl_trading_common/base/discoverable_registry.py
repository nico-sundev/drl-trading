import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TypeVar, Generic, Any

logger = logging.getLogger(__name__)

# Generic type variables for the base class types
T = TypeVar('T')  # The type of class being registered (e.g., BaseFeature, BaseParameterSetConfig)
K = TypeVar('K')  # The type of key being used (e.g., str, Enum)


class DiscoverableRegistry(Generic[K, T], ABC):
    """
    Abstract base class for discoverable registries.

    This class provides common functionality for registries that need to:
    - Store and retrieve class types by key (string or enum)
    - Discover classes from packages automatically
    - Provide basic CRUD operations for class management

    Type Parameters:
        T: The type of class being registered (e.g., BaseFeature, BaseParameterSetConfig)
        K: The type of key being used (e.g., str, Enum)

    Subclasses should implement the abstract methods to define:
    - How to validate classes during registration
    - How to extract keys from class names
    - What criteria to use for auto-discovery
    """

    def __init__(self) -> None:
        self._class_map: Dict[K, Type[T]] = {}

    def get_class(self, key: K) -> Optional[Type[T]]:
        """
        Get a class for a given key.

        Args:
            key: The key to look up

        Returns:
            The class if found, None otherwise
        """
        return self._class_map.get(key)

    def register_class(self, key: K, class_type: Type[T]) -> None:
        """
        Register a class for a given key.

        Args:
            key: The key to register
            class_type: The class to register

        Raises:
            TypeError: If class validation fails
        """
        self._validate_class(class_type)

        if key in self._class_map:
            self._handle_duplicate_registration(key, class_type)

        self._class_map[key] = class_type
        logger.debug(f"Registered class '{key}': {class_type}")

    def discover_classes(self, package_name: str) -> Dict[K, Type[T]]:
        """
        Discover and register classes from a specified package.

        Args:
            package_name: The name of the package to discover classes from

        Returns:
            A dictionary mapping keys to their corresponding class types
        """
        logger.info(f"Starting class discovery from package: {package_name}")
        discovered_count = 0
        processed_modules = 0

        if not package_name or not package_name.strip():
            raise ValueError("Package name for class discovery must be a non-empty string.")

        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error(f"Could not import package {package_name}: {e}")
            return self._class_map

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                continue

            full_module_name = f"{package_name}.{module_name}"
            processed_modules += 1

            try:
                module = importlib.import_module(full_module_name)
            except ImportError as e:
                logger.warning(f"Failed to import module {full_module_name}: {e}")
                continue

            module_classes_found = 0
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._should_discover_class(obj):
                    key = self._extract_key_from_class(obj)
                    logger.debug(
                        f"Discovered class: '{key}' from class {name} "
                        f"in module {full_module_name}"
                    )
                    self.register_class(key, obj)
                    discovered_count += 1
                    module_classes_found += 1

            if module_classes_found > 0:
                logger.debug(
                    f"Found {module_classes_found} class(es) in module {full_module_name}"
                )

        logger.info(
            f"Class discovery complete. Found {discovered_count} classes across {processed_modules} modules"
        )
        return self._class_map

    def reset(self) -> None:
        """Clear all registered classes and reset the registry state."""
        logger.debug(f"Resetting {self.__class__.__name__}")
        self._class_map.clear()

    @abstractmethod
    def _validate_class(self, class_type: Type[T]) -> None:
        """
        Validate that a class meets the requirements for this registry.

        Args:
            class_type: The class to validate

        Raises:
            TypeError: If the class doesn't meet requirements
        """
        pass

    @abstractmethod
    def _should_discover_class(self, class_obj: Any) -> bool:
        """
        Determine if a class should be discovered and registered.

        Args:
            class_obj: The class object to check

        Returns:
            True if the class should be discovered, False otherwise
        """
        pass

    @abstractmethod
    def _extract_key_from_class(self, class_obj: Any) -> K:
        """
        Extract the registry key from a class object and name.

        Args:
            class_obj: The class object
            class_name: The full class name

        Returns:
            The extracted key for registry purposes
        """
        pass

    def _handle_duplicate_registration(self, key: K, new_class_type: Type[T]) -> None:
        """
        Handle the case where a class is being registered for an existing key.
        Default implementation logs a warning. Subclasses can override for different behavior.

        Args:
            key: The registry key
            new_class_type: The new class being registered
        """
        logger.warning(
            f"Overriding existing class for key '{key}': "
            f"{self._class_map[key].__name__} -> {new_class_type.__name__}"
        )
