"""
Resampling context for managing stateful multi-symbol/timeframe processing.

This module provides state management for the resampling service, tracking
last processed timestamps, accumulator states, and enabling incremental
processing with service restart recovery.
"""

from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.resample.timeframe_candle_accumulator import TimeframeCandleAccumulator


@dataclass
class SymbolTimeframeState:
    """
    State tracking for a specific symbol/timeframe combination.

    This tracks the processing state needed for incremental resampling
    and accumulator persistence across service restarts.
    """
    symbol: str
    timeframe: Timeframe
    last_processed_timestamp: Optional[datetime] = None
    records_processed: int = 0
    candles_generated: int = 0
    last_updated: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize last_updated if not provided."""
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class AccumulatorState:
    """
    Serializable state of a candle accumulator.

    This allows accumulator state to be persisted and restored
    across service restarts, enabling true incremental processing.
    """
    symbol: str
    target_timeframe: Timeframe
    current_period_start: Optional[datetime] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: int = 0
    record_count: int = 0

    def to_accumulator(self) -> TimeframeCandleAccumulator:
        """Convert to active TimeframeCandleAccumulator."""
        accumulator = TimeframeCandleAccumulator(timeframe=self.target_timeframe)
        accumulator.current_period_start = self.current_period_start
        accumulator.open_price = self.open_price
        accumulator.high_price = self.high_price
        accumulator.low_price = self.low_price
        accumulator.close_price = self.close_price
        accumulator.volume = self.volume
        accumulator.record_count = self.record_count
        return accumulator

    @classmethod
    def from_accumulator(
        cls,
        accumulator: TimeframeCandleAccumulator,
        symbol: str
    ) -> "AccumulatorState":
        """Create from active TimeframeCandleAccumulator."""
        return cls(
            symbol=symbol,
            target_timeframe=accumulator.timeframe,
            current_period_start=accumulator.current_period_start,
            open_price=accumulator.open_price,
            high_price=accumulator.high_price,
            low_price=accumulator.low_price,
            close_price=accumulator.close_price,
            volume=accumulator.volume,
            record_count=accumulator.record_count
        )


class ResamplingContext:
    """
    Context manager for stateful resampling operations.

    Manages state for multiple symbols and timeframes, enabling:
    - Incremental processing from last processed timestamps
    - Accumulator state persistence across service restarts
    - Memory-efficient processing with configurable limits
    - Service restart recovery
    """

    def __init__(self, max_symbols_in_memory: int = 100):
        """
        Initialize resampling context.

        Args:
            max_symbols_in_memory: Maximum symbols to keep in memory simultaneously
        """
        self.max_symbols_in_memory = max_symbols_in_memory

        # State tracking
        self._symbol_states: Dict[str, Dict[Timeframe, SymbolTimeframeState]] = {}
        self._accumulator_states: Dict[str, Dict[Timeframe, AccumulatorState]] = {}

        # Active processing context
        self._active_accumulators: Dict[str, Dict[Timeframe, TimeframeCandleAccumulator]] = {}

    @property
    def symbol_states(self) -> Dict[str, Dict[Timeframe, SymbolTimeframeState]]:
        """Public access to symbol states for serialization."""
        return self._symbol_states

    def get_last_processed_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe
    ) -> Optional[datetime]:
        """
        Get the last processed timestamp for symbol/timeframe.

        Used to determine start_time for incremental database queries.
        """
        if symbol not in self._symbol_states:
            return None

        timeframe_states = self._symbol_states[symbol]
        if timeframe not in timeframe_states:
            return None

        return timeframe_states[timeframe].last_processed_timestamp

    def update_last_processed_timestamp(
        self,
        symbol: str,
        timeframe: Timeframe,
        timestamp: datetime
    ) -> None:
        """Update the last processed timestamp for symbol/timeframe."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = {}

        if timeframe not in self._symbol_states[symbol]:
            self._symbol_states[symbol][timeframe] = SymbolTimeframeState(
                symbol=symbol,
                timeframe=timeframe
            )

        state = self._symbol_states[symbol][timeframe]
        state.last_processed_timestamp = timestamp
        state.last_updated = datetime.now()

    def get_accumulator(
        self,
        symbol: str,
        target_timeframe: Timeframe
    ) -> TimeframeCandleAccumulator:
        """
        Get or create accumulator for symbol/target_timeframe.

        Restores from persisted state if available.
        """
        if symbol not in self._active_accumulators:
            self._active_accumulators[symbol] = {}

        symbol_accumulators = self._active_accumulators[symbol]

        if target_timeframe not in symbol_accumulators:
            # Try to restore from persisted state
            if (symbol in self._accumulator_states and
                target_timeframe in self._accumulator_states[symbol]):

                persisted_state = self._accumulator_states[symbol][target_timeframe]
                symbol_accumulators[target_timeframe] = persisted_state.to_accumulator()
            else:
                # Create new accumulator
                symbol_accumulators[target_timeframe] = TimeframeCandleAccumulator(
                    timeframe=target_timeframe
                )

        return symbol_accumulators[target_timeframe]

    def persist_accumulator_state(
        self,
        symbol: str,
        accumulator: TimeframeCandleAccumulator
    ) -> None:
        """Persist accumulator state for service restart recovery."""
        if symbol not in self._accumulator_states:
            self._accumulator_states[symbol] = {}

        accumulator_state = AccumulatorState.from_accumulator(accumulator, symbol)
        self._accumulator_states[symbol][accumulator.timeframe] = accumulator_state

    def get_symbols_for_processing(self) -> List[str]:
        """Get list of symbols that have active state."""
        return list(self._symbol_states.keys())

    def clean_inactive_symbols(self, active_symbols: List[str]) -> None:
        """Remove state for symbols not in active list to manage memory."""
        all_symbols = set(self._symbol_states.keys())
        inactive_symbols = all_symbols - set(active_symbols)

        for symbol in inactive_symbols:
            if symbol in self._symbol_states:
                del self._symbol_states[symbol]
            if symbol in self._accumulator_states:
                del self._accumulator_states[symbol]
            if symbol in self._active_accumulators:
                del self._active_accumulators[symbol]

    def get_processing_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get processing statistics for monitoring."""
        stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        for symbol, timeframe_states in self._symbol_states.items():
            stats[symbol] = {}
            for timeframe, state in timeframe_states.items():
                stats[symbol][timeframe.value] = {
                    'records_processed': state.records_processed,
                    'candles_generated': state.candles_generated
                }

        return stats

    def increment_stats(
        self,
        symbol: str,
        timeframe: Timeframe,
        records_processed: int = 0,
        candles_generated: int = 0
    ) -> None:
        """Increment processing statistics."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = {}

        if timeframe not in self._symbol_states[symbol]:
            self._symbol_states[symbol][timeframe] = SymbolTimeframeState(
                symbol=symbol,
                timeframe=timeframe
            )

        state = self._symbol_states[symbol][timeframe]
        state.records_processed += records_processed
        state.candles_generated += candles_generated
        state.last_updated = datetime.now()
