"""
Market data resampling service for generating higher timeframe OHLCV candles.

This service provides memory-efficient streaming resampling of market data
from lower timeframes to higher timeframes using a multi-pointer approach
for optimal O(n) complexity.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from injector import inject

from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_preprocess.core.model.resample.resampling_response import ResamplingResponse
from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext
from drl_trading_preprocess.core.port.message_publisher_port import MessagePublisherPort
from drl_trading_preprocess.core.port.state_persistence_port import IStatePersistencePort
from drl_trading_preprocess.core.service.resample.candle_accumulator_service import CandleAccumulatorService
from drl_trading_preprocess.infrastructure.config.preprocess_config import ResampleConfig
from drl_trading_preprocess.infrastructure.adapter.state_persistence.noop_state_persistence_service import NoOpStatePersistenceService


@inject
class MarketDataResamplingService:
    """
    Stateful service for resampling market data to higher timeframes.

    Provides memory-efficient incremental resampling with persistent state
    management, supporting service restarts and Kafka-driven processing.
    """

    def __init__(
        self,
        market_data_reader: MarketDataReaderPort,
        message_publisher: MessagePublisherPort,
        candle_accumulator_service: CandleAccumulatorService,
        resample_config: ResampleConfig,
        state_persistence: IStatePersistencePort
    ):
        """
        Initialize resampling service with required dependencies.

        Args:
            market_data_reader: Port for reading market data from repository
            message_publisher: Port for publishing resampled data to messaging
            candle_accumulator_service: Service for OHLCV aggregation logic
            resample_config: Configuration for resampling operations
            state_persistence: Optional state persistence service (injected conditionally)
        """
        self.market_data_reader = market_data_reader
        self.message_publisher = message_publisher
        self.candle_accumulator_service = candle_accumulator_service
        self.resample_config = resample_config
        # Use NoOpStatePersistenceService directly to leverage the Null Object pattern
        self.state_persistence = state_persistence
        
        self.logger = logging.getLogger(__name__)

        # Initialize or restore resampling context
        self.context = self._initialize_context()

        self.logger.info(
            f"MarketDataResamplingService initialized with "
            f"state persistence={'enabled' if self.state_persistence else 'disabled'}, "
            f"context symbols={len(self.context.get_symbols_for_processing())}"
        )

    def _initialize_context(self) -> ResamplingContext:
        """Initialize or restore resampling context from persistent storage.

        Returns:
            Initialized ResamplingContext (either restored from disk or new)
        """
        # Try to restore from persistent storage if enabled
        if self.state_persistence:
            restored_context = self.state_persistence.load_context()
            if restored_context:
                self.logger.info("Restored resampling context from persistent storage")
                return restored_context

        # Create new context with configuration
        max_symbols = getattr(self.resample_config, 'max_symbols_in_memory', 100)
        context = ResamplingContext(max_symbols_in_memory=max_symbols)
        self.logger.info("Created new resampling context")
        return context

    def _save_context_if_enabled(self) -> None:
        """Save context to persistent storage if enabled."""
        if self.state_persistence:
            self.state_persistence.auto_save_if_needed(self.context)

    def resample_symbol_data_incremental(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        target_timeframes: List[Timeframe]
    ) -> ResamplingResponse:
        """
        Perform incremental resampling for a symbol using persistent state.

        This is the primary method for Kafka-driven incremental processing.
        Uses last processed timestamps to fetch only new data, enabling
        efficient processing and service restart recovery.

        Args:
            symbol: Trading symbol to resample
            base_timeframe: Source timeframe for resampling
            target_timeframes: List of target timeframes to generate

        Returns:
            ResamplingResponse: Complete resampling results
        """
        processing_start = datetime.now()

        try:
            self.logger.info(
                f"Starting incremental resampling for {symbol}: "
                f"{base_timeframe.value} → {[tf.value for tf in target_timeframes]}"
            )

            # Get last processed timestamp for incremental fetch
            last_timestamp = self.context.get_last_processed_timestamp(symbol, base_timeframe)
            start_time = last_timestamp or self.resample_config.historical_start_date
            end_time = datetime.now()

            # Fetch data incrementally using pagination
            base_data = self._get_base_data_paginated(
                symbol, base_timeframe, start_time, end_time
            )

            if not base_data:
                self.logger.info(f"No new data for {symbol} since {last_timestamp}")
                return self._create_empty_response(symbol, base_timeframe, processing_start)

            # Process data with stateful accumulators
            resampled_data, new_candles_count = self._process_incremental_data(
                symbol, base_data, target_timeframes
            )

            # Update state tracking
            latest_timestamp = max(record.timestamp for record in base_data)
            self.context.update_last_processed_timestamp(symbol, base_timeframe, latest_timestamp)

            processing_end = datetime.now()

            # Create response
            response = ResamplingResponse(
                symbol=symbol,
                base_timeframe=base_timeframe,
                resampled_data=resampled_data,
                new_candles_count=new_candles_count,
                processing_start_time=processing_start,
                processing_end_time=processing_end,
                source_records_processed=len(base_data)
            )

            # Publish results
            self.message_publisher.publish_resampled_data(
                symbol=symbol,
                base_timeframe=base_timeframe,
                resampled_data=resampled_data,
                new_candles_count=new_candles_count
            )

            self.logger.info(
                f"Completed incremental resampling for {symbol} in {response.processing_duration_ms}ms: "
                f"{len(base_data)} source records → {response.total_new_candles} new candles"
            )

            # Save context state if persistence is enabled
            self._save_context_if_enabled()

            return response

        except Exception as e:
            processing_end = datetime.now()
            error_message = f"Incremental resampling failed for {symbol}: {str(e)}"

            self.logger.error(error_message, exc_info=True)

            # Save context state even on error (to preserve partial progress)
            self._save_context_if_enabled()

            # Publish error
            self.message_publisher.publish_resampling_error(
                symbol=symbol,
                base_timeframe=base_timeframe,
                target_timeframes=target_timeframes,
                error_message=error_message,
                error_details={
                    "error_type": type(e).__name__,
                    "processing_duration_ms": str(
                        int((processing_end - processing_start).total_seconds() * 1000)
                    )
                }
            )

            raise ValueError(error_message)

    def _get_base_data_paginated(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataModel]:
        """
        Get base timeframe data using pagination to avoid memory issues.

        Handles both incremental updates and cold starts by fetching data
        in configurable chunks from the database.
        """
        chunk_size = getattr(self.resample_config, 'pagination_limit', 10000)
        all_data = []
        offset = 0

        self.logger.debug(
            f"Fetching {symbol} data from {start_time} to {end_time} "
            f"(chunk_size={chunk_size})"
        )

        while True:
            try:
                # Use paginated data access
                chunk = self.market_data_reader.get_symbol_data_range_paginated(
                    symbol=symbol,
                    timeframe=base_timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    limit=chunk_size,
                    offset=offset
                )

                if not chunk:
                    break

                # Simple data quality validation (no complex buffering)
                valid_chunk = [record for record in chunk if not self._is_invalid_ohlcv(record)]
                all_data.extend(valid_chunk)

                self.logger.debug(
                    f"Fetched chunk: {len(chunk)} records, "
                    f"valid: {len(valid_chunk)}, total: {len(all_data)}"
                )

                # If we got less than chunk_size, we're done
                if len(chunk) < chunk_size:
                    break

                offset += chunk_size

            except AttributeError:
                # Fallback to non-paginated method if not implemented
                self.logger.warning(
                    f"Pagination not available, falling back to full range query for {symbol}"
                )
                return self.market_data_reader.get_symbol_data_range(
                    symbol=symbol,
                    timeframe=base_timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

        self.logger.info(f"Fetched {len(all_data)} records for {symbol} ({base_timeframe.value})")
        return all_data

    def _process_incremental_data(
        self,
        symbol: str,
        base_data: List[MarketDataModel],
        target_timeframes: List[Timeframe]
    ) -> tuple[Dict[Timeframe, List[MarketDataModel]], Dict[Timeframe, int]]:
        """
        Process base data using stateful accumulators.

        Much simpler than the complex streaming approach - uses the context
        to maintain accumulator state across processing cycles.
        """
        resampled_data: Dict[Timeframe, List[MarketDataModel]] = {}
        new_candles_count: Dict[Timeframe, int] = {}

        # Initialize result containers
        for tf in target_timeframes:
            resampled_data[tf] = []
            new_candles_count[tf] = 0

        # Process each record
        for record in base_data:
            for target_timeframe in target_timeframes:
                # Get or restore accumulator from context
                accumulator = self.context.get_accumulator(symbol, target_timeframe)

                # Check if new period starts
                if self.candle_accumulator_service.should_start_new_period(accumulator, record):
                    # Emit completed candle
                    if not accumulator.is_empty():
                        completed_candle = self.candle_accumulator_service.build_candle_from_accumulator(
                            accumulator, symbol
                        )
                        resampled_data[target_timeframe].append(completed_candle)
                        new_candles_count[target_timeframe] += 1

                    # Reset for new period
                    accumulator.reset()

                # Add record to accumulator
                self.candle_accumulator_service.add_record_to_accumulator(accumulator, record)

                # Persist accumulator state
                self.context.persist_accumulator_state(symbol, accumulator)

        # Emit any remaining incomplete candles at the end (if enabled in config)
        if self.resample_config.enable_incomplete_candle_publishing:
            for target_timeframe in target_timeframes:
                accumulator = self.context.get_accumulator(symbol, target_timeframe)
                if not accumulator.is_empty():
                    # Build final candle from remaining data
                    final_candle = self.candle_accumulator_service.build_candle_from_accumulator(
                        accumulator, symbol
                    )
                    resampled_data[target_timeframe].append(final_candle)
                    new_candles_count[target_timeframe] += 1

                    # Keep accumulator state for next incremental processing
                    # (Don't reset, so incomplete period continues in next batch)

        # Update stats
        for target_timeframe in target_timeframes:
            self.context.increment_stats(
                symbol,
                target_timeframe,
                records_processed=len(base_data),
                candles_generated=new_candles_count[target_timeframe]
            )

        return resampled_data, new_candles_count

    def _create_empty_response(
        self,
        symbol: str,
        base_timeframe: Timeframe,
        processing_start: datetime
    ) -> ResamplingResponse:
        """Create response for cases with no new data."""
        return ResamplingResponse(
            symbol=symbol,
            base_timeframe=base_timeframe,
            resampled_data={},
            new_candles_count={},
            processing_start_time=processing_start,
            processing_end_time=datetime.now(),
            source_records_processed=0
        )

    def get_existing_data_summary(
        self,
        symbol: str,
        timeframes: List[Timeframe]
    ) -> Dict[Timeframe, Optional[datetime]]:
        """
        Get summary of existing data for incremental processing decisions.

        Returns the latest timestamp for each timeframe, helping determine
        what incremental processing is needed.
        """
        summary: Dict[Timeframe, Optional[datetime]] = {}

        for timeframe in timeframes:
            try:
                latest_data = self.market_data_reader.get_latest_prices([symbol], timeframe)
                if latest_data:
                    summary[timeframe] = latest_data[0].timestamp
                else:
                    summary[timeframe] = None
            except Exception as e:
                self.logger.warning(
                    f"Could not get latest timestamp for {symbol}:{timeframe.value}: {e}"
                )
                summary[timeframe] = None

        return summary

    def get_processing_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get processing statistics from the resampling context."""
        return self.context.get_processing_stats()

    def reset_symbol_state(self, symbol: str) -> None:
        """Reset processing state for a symbol (useful for reprocessing)."""
        self.context.clean_inactive_symbols([])  # Remove all for this symbol
        self.logger.info(f"Reset processing state for {symbol}")

    def get_symbols_in_context(self) -> List[str]:
        """Get list of symbols currently being tracked."""
        return self.context.get_symbols_for_processing()

    def save_context_state(self) -> bool:
        """Manually save the resampling context state.

        Returns:
            True if save was successful, False otherwise
        """
        if self.state_persistence:
            return self.state_persistence.save_context(self.context)
        else:
            self.logger.warning("State persistence is not enabled")
            return False

    def reset_context_state(self) -> bool:
        """Reset and cleanup the resampling context state.

        Returns:
            True if context was reset successfully, False if persistent cleanup failed
        """
        # Clear in-memory context
        self.context = ResamplingContext(
            max_symbols_in_memory=getattr(self.resample_config, 'max_symbols_in_memory', 100)
        )

        # Clean up persistent state if enabled
        if self.state_persistence:
            success = self.state_persistence.cleanup_state_file()
            self.logger.info("Reset resampling context and cleaned up persistent state")
            return success
        else:
            # No persistent state to clean, but in-memory reset succeeded
            self.logger.info("Reset resampling context (no persistent state to clean)")
            return True

    def _is_invalid_ohlcv(self, record: MarketDataModel) -> bool:
        """
        Check if a market data record has invalid OHLCV relationships.

        Invalid conditions:
        - High price < Low price
        - Open/Close outside High/Low range
        - Negative volume
        - Zero or negative prices

        Args:
            record: Market data record to validate

        Returns:
            True if the record has invalid OHLCV data
        """
        try:
            # Check for negative or zero prices
            if (record.open_price <= 0 or record.high_price <= 0 or
                record.low_price <= 0 or record.close_price <= 0):
                return True

            # Check high < low (impossible market condition)
            if record.high_price < record.low_price:
                return True

            # Check if open/close are outside high/low range
            if (record.open_price > record.high_price or record.open_price < record.low_price or
                record.close_price > record.high_price or record.close_price < record.low_price):
                return True

            # Check for negative volume
            if record.volume < 0:
                return True

            return False

        except (AttributeError, TypeError):
            # If any price/volume field is missing or invalid type
            return True
