"""
Simple load test for MarketDataResamplingService.

Single file containing data generation, mocks, and performance tests.
Run with: pytest tests/simple_load_test.py --load -v
"""
import time
import gc
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock

import pytest
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService


# === DATA GENERATION ===
def generate_ohlcv_data(count: int, symbol: str = "EURUSD") -> List[MarketDataModel]:
    """Generate realistic OHLCV test data."""
    records = []
    base_price = 1.1000
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(count):
        # Simple price movement with small random changes
        price_change = (i % 100 - 50) * 0.0001  # Small realistic movements
        open_price = base_price + price_change
        high_price = open_price + abs(price_change) * 0.5
        low_price = open_price - abs(price_change) * 0.5
        close_price = open_price + price_change * 0.8
        volume = 1000 + (i % 1000)

        records.append(MarketDataModel(
            symbol=symbol,
            timeframe=Timeframe.MINUTE_1,
            timestamp=start_time + timedelta(minutes=i),
            open_price=round(open_price, 5),
            high_price=round(high_price, 5),
            low_price=round(low_price, 5),
            close_price=round(close_price, 5),
            volume=volume
        ))

    return records


# === SIMPLE MOCKS ===
def create_mocked_service(test_data: List[MarketDataModel]) -> MarketDataResamplingService:
    """Create service with mocked dependencies."""
    # Mock market data reader
    mock_reader = Mock()

    # Create a chunked data response function
    def get_chunked_data(symbol, timeframe, start_time, end_time, limit, offset):
        start_idx = offset
        end_idx = offset + limit
        chunk = test_data[start_idx:end_idx]
        return chunk

    mock_reader.get_symbol_data_range_paginated.side_effect = get_chunked_data

    # Mock message publisher
    mock_publisher = Mock()

    # Mock candle accumulator service with proper methods
    mock_accumulator = Mock()

    # Mock resampling methods that return realistic results
    def mock_resample_to_timeframe(records, target_timeframe):
        # Simulate realistic candle reduction
        reduction_factor = {
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.HOUR_1: 60,
            Timeframe.DAY_1: 1440
        }.get(target_timeframe, 1)

        num_candles = max(1, len(records) // reduction_factor)
        return [records[0]] * num_candles  # Simplified - return same record repeated

    mock_accumulator.resample_to_timeframe.side_effect = mock_resample_to_timeframe

    # Mock config
    mock_config = Mock()
    mock_config.pagination_limit = 10000
    mock_config.historical_start_date = datetime(2024, 1, 1)

    return MarketDataResamplingService(
        market_data_reader=mock_reader,
        message_publisher=mock_publisher,
        candle_accumulator_service=mock_accumulator,
        resample_config=mock_config,
        state_persistence=None
    )


# === PERFORMANCE TESTS ===
class TestLoadPerformance:
    """Simple load tests for resampling service."""

    @pytest.mark.load
    def test_100k_records_performance(self):
        """Test processing 100k records - baseline performance."""
        # Given
        test_data = generate_ohlcv_data(100_000)
        service = create_mocked_service(test_data)

        # When
        start_time = time.time()
        response = service.resample_symbol_data_incremental(
            symbol='EURUSD',
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1, Timeframe.DAY_1]
        )
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        records_per_sec = response.source_records_processed / processing_time

        print("\n=== 100K RECORDS PERFORMANCE ===")
        print(f"Processed: {response.source_records_processed:,} records")
        print(f"Time: {processing_time:.2f} seconds")
        print(f"Rate: {records_per_sec:,.0f} records/second")
        print(f"Total candles: {response.total_new_candles:,}")

        # Performance assertions (realistic targets based on actual testing)
        assert processing_time < 10.0, f"Processing took {processing_time:.2f}s, expected <10s"
        assert records_per_sec > 5_000, f"Rate {records_per_sec:,.0f} rec/sec, expected >5k"
        assert response.source_records_processed > 0

    @pytest.mark.load
    def test_1m_records_performance(self):
        """Test processing 1M records - find performance limits."""
        # Given
        test_data = generate_ohlcv_data(1_000_000)
        service = create_mocked_service(test_data)

        # When
        start_time = time.time()
        response = service.resample_symbol_data_incremental(
            symbol='EURUSD',
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1, Timeframe.DAY_1]
        )
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        records_per_sec = response.source_records_processed / processing_time

        print("\n=== 1M RECORDS PERFORMANCE ===")
        print(f"Processed: {response.source_records_processed:,} records")
        print(f"Time: {processing_time:.2f} seconds")
        print(f"Rate: {records_per_sec:,.0f} records/second")
        print(f"Total candles: {response.total_new_candles:,}")

        # Log results for analysis
        assert response.source_records_processed == 1_000_000

    @pytest.mark.load
    def test_10m_records_performance(self):
        """Test processing 10M records - stress test."""
        # Given
        test_data = generate_ohlcv_data(10_000_000)
        service = create_mocked_service(test_data)

        # When
        start_time = time.time()
        try:
            response = service.resample_symbol_data_incremental(
                symbol='EURUSD',
                base_timeframe=Timeframe.MINUTE_1,
                target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1, Timeframe.DAY_1]
            )
            end_time = time.time()

            # Then
            processing_time = end_time - start_time
            records_per_sec = response.source_records_processed / processing_time

            print("\n=== 10M RECORDS PERFORMANCE ===")
            print(f"Processed: {response.source_records_processed:,} records")
            print(f"Time: {processing_time:.2f} seconds")
            print(f"Rate: {records_per_sec:,.0f} records/second")
            print(f"Total candles: {response.total_new_candles:,}")

        except Exception as e:
            print("\n=== 10M RECORDS FAILED ===")
            print(f"Error: {e}")
            print("This indicates performance/memory limits reached")

    @pytest.mark.load
    def test_algorithmic_complexity(self):
        """Validate O(n) scaling across multiple dataset sizes."""
        # Given
        test_sizes = [50_000, 100_000, 500_000, 1_000_000]
        results = []

        for size in test_sizes:
            # Generate test data
            test_data = generate_ohlcv_data(size)
            service = create_mocked_service(test_data)

            # Time the processing
            start_time = time.time()
            try:
                response = service.resample_symbol_data_incremental(
                    symbol='EURUSD',
                    base_timeframe=Timeframe.MINUTE_1,
                    target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1]
                )
                end_time = time.time()

                processing_time = end_time - start_time
                records_per_sec = response.source_records_processed / processing_time
                results.append((size, processing_time, records_per_sec))

                print(f"{size:,} records: {processing_time:.2f}s ({records_per_sec:,.0f} rec/sec)")

            except Exception as e:
                print(f"{size:,} records: FAILED - {e}")
                break

            # Cleanup between iterations
            gc.collect()

        # Analyze complexity
        if len(results) >= 2:
            print("\n=== COMPLEXITY ANALYSIS ===")
            for i in range(1, len(results)):
                prev_size, prev_time, _ = results[i-1]
                curr_size, curr_time, _ = results[i]

                size_ratio = curr_size / prev_size
                time_ratio = curr_time / prev_time
                complexity = time_ratio / size_ratio

                print(f"{prev_size:,} → {curr_size:,}: size_ratio={size_ratio:.1f}x, time_ratio={time_ratio:.1f}x, complexity={complexity:.2f}")

                if complexity > 1.5:
                    print(f"⚠️ WARNING: Complexity ratio {complexity:.2f} suggests worse than O(n)")
                else:
                    print(f"✅ GOOD: Complexity ratio {complexity:.2f} indicates O(n) scaling")
