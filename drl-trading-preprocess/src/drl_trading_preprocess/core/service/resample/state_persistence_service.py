"""State persistence service for ResamplingContext."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext


class StatePersistenceService:
    """Service for persisting and restoring ResamplingContext state.

    This service handles saving and loading of resampling context state
    to/from the filesystem, enabling service restart recovery and
    persistent accumulator state management.
    """

    def __init__(self, state_file_path: str, backup_interval: int = 1000) -> None:
        """Initialize the state persistence service.

        Args:
            state_file_path: Path to the state persistence file
            backup_interval: Number of operations between automatic state saves
        """
        self.state_file_path = Path(state_file_path)
        self.backup_interval = backup_interval
        self.operation_count = 0
        self.logger = logging.getLogger(__name__)

        # Ensure the state directory exists
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_context(self, context: ResamplingContext) -> bool:
        """Save ResamplingContext state to persistent storage.

        Args:
            context: The resampling context to save

        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create backup of existing state file if it exists
            if self.state_file_path.exists():
                backup_path = self.state_file_path.with_suffix('.json.backup')
                self.state_file_path.rename(backup_path)
                self.logger.debug(f"Created backup at {backup_path}")

            # Serialize context to JSON
            state_data = {
                'saved_at': datetime.now().isoformat(),
                'max_symbols_in_memory': context.max_symbols_in_memory,
                'symbol_states': {}
            }

            # Save symbol states
            for symbol, timeframe_states in context.symbol_states.items():
                state_data['symbol_states'][symbol] = {}  # type: ignore
                for timeframe, symbol_state in timeframe_states.items():
                    state_data['symbol_states'][symbol][timeframe.value] = {  # type: ignore
                        'symbol': symbol_state.symbol,
                        'timeframe': symbol_state.timeframe.value,
                        'last_processed_timestamp': symbol_state.last_processed_timestamp.isoformat() if symbol_state.last_processed_timestamp else None,
                        'records_processed': symbol_state.records_processed,
                        'candles_generated': symbol_state.candles_generated,
                        'last_updated': symbol_state.last_updated.isoformat() if symbol_state.last_updated else None
                    }

            # Write to file
            with open(self.state_file_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            self.logger.info(f"Saved resampling context state with {len(context.symbol_states)} symbols to {self.state_file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save resampling context state: {e}")
            return False

    def load_context(self) -> Optional[ResamplingContext]:
        """Load ResamplingContext state from persistent storage.

        Returns:
            Restored ResamplingContext or None if no valid state file exists
        """
        if not self.state_file_path.exists():
            self.logger.info(f"No state file found at {self.state_file_path}, starting with empty context")
            return None

        try:
            with open(self.state_file_path, 'r') as f:
                state_data = json.load(f)

            # Create new context
            max_symbols = state_data.get('max_symbols_in_memory', 100)
            context = ResamplingContext(max_symbols_in_memory=max_symbols)

            # Restore symbol states
            from drl_trading_preprocess.core.model.resample.resampling_context import SymbolTimeframeState
            from drl_trading_common.model.timeframe import Timeframe

            for symbol, timeframe_data in state_data.get('symbol_states', {}).items():
                context._symbol_states[symbol] = {}
                for timeframe_str, state_data_item in timeframe_data.items():
                    timeframe = Timeframe(state_data_item['timeframe'])
                    symbol_state = SymbolTimeframeState(
                        symbol=state_data_item['symbol'],
                        timeframe=timeframe,
                        last_processed_timestamp=(
                            datetime.fromisoformat(state_data_item['last_processed_timestamp'])
                            if state_data_item.get('last_processed_timestamp') else None
                        ),
                        records_processed=state_data_item.get('records_processed', 0),
                        candles_generated=state_data_item.get('candles_generated', 0),
                        last_updated=(
                            datetime.fromisoformat(state_data_item['last_updated'])
                            if state_data_item.get('last_updated') else datetime.now()
                        )
                    )
                    context._symbol_states[symbol][timeframe] = symbol_state

            saved_at = state_data.get('saved_at')
            self.logger.info(
                f"Loaded resampling context state with {len(context.symbol_states)} symbols "
                f"(saved at {saved_at})"
            )
            return context

        except Exception as e:
            self.logger.error(f"Failed to load resampling context state: {e}")
            # Try to load backup if available
            backup_path = self.state_file_path.with_suffix('.json.backup')
            if backup_path.exists():
                self.logger.info(f"Attempting to load backup from {backup_path}")
                try:
                    self.state_file_path = backup_path
                    return self.load_context()
                except Exception as backup_error:
                    self.logger.error(f"Failed to load backup state: {backup_error}")

            return None

    def auto_save_if_needed(self, context: ResamplingContext) -> None:
        """Automatically save context if backup interval is reached.

        Args:
            context: The resampling context to potentially save
        """
        self.operation_count += 1
        if self.operation_count >= self.backup_interval:
            self.save_context(context)
            self.operation_count = 0

    def cleanup_state_file(self) -> bool:
        """Remove the state file and backup.

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if self.state_file_path.exists():
                self.state_file_path.unlink()
                self.logger.info(f"Removed state file {self.state_file_path}")

            backup_path = self.state_file_path.with_suffix('.json.backup')
            if backup_path.exists():
                backup_path.unlink()
                self.logger.info(f"Removed backup file {backup_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to cleanup state files: {e}")
            return False
