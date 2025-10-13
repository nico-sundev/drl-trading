"""Port interface for state persistence functionality."""

from abc import ABC, abstractmethod
from typing import Optional

from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext


class IStatePersistencePort(ABC):
    """
    Interface for persisting and restoring ResamplingContext state.

    This port enables the resampling service to persist state between restarts
    without being coupled to specific persistence implementations.

    Implementations:
    - StatePersistenceService: File-based persistence for production
    - NoOpStatePersistenceService: No-op implementation when persistence is disabled
    """

    @abstractmethod
    def save_context(self, context: ResamplingContext) -> bool:
        """
        Save ResamplingContext state to persistent storage.

        Args:
            context: The resampling context to save

        Returns:
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def load_context(self) -> Optional[ResamplingContext]:
        """
        Load ResamplingContext state from persistent storage.

        Returns:
            Restored ResamplingContext or None if no valid state exists
        """
        pass

    @abstractmethod
    def auto_save_if_needed(self, context: ResamplingContext) -> None:
        """
        Automatically save context if backup interval is reached.

        Args:
            context: The resampling context to potentially save
        """
        pass

    @abstractmethod
    def cleanup_state_file(self) -> bool:
        """
        Remove the state file and backup.

        Returns:
            True if cleanup was successful, False otherwise
        """
        pass
