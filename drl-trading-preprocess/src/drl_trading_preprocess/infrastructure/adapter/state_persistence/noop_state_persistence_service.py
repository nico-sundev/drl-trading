"""No-op implementation of state persistence service."""

import logging
from typing import Optional

from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext
from drl_trading_preprocess.core.port.state_persistence_port import IStatePersistencePort


class NoOpStatePersistenceService(IStatePersistencePort):
    """No-operation state persistence service.

    This implementation safely does nothing when state persistence is disabled.
    It implements the Null Object Pattern to avoid conditional logic in clients.
    """

    def __init__(self) -> None:
        """Initialize no-op state persistence service."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("NoOpStatePersistenceService initialized - state persistence is disabled")

    def save_context(self, context: ResamplingContext) -> bool:
        """No-op save operation.

        Args:
            context: Resampling context (ignored)
        """
        return False

    def load_context(self) -> Optional[ResamplingContext]:
        """No-op load operation.

        Returns:
            None (no state to load)
        """
        return None

    def auto_save_if_needed(self, context: ResamplingContext) -> None:
        """No-op auto-save operation.

        Args:
            context: Resampling context (ignored)
            current_candle_count: Current candle count (ignored)
        """
        pass

    def cleanup_state_file(self) -> bool:
        """No-op cleanup operation."""
        return False
