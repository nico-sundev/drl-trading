"""
Simplified comprehensive load test for MarketDataResamplingService.

Single file testing approach to find actual performance limits.
Run with: pytest tests/comprehensive_load_test.py -m load -v -s
"""
import time
import gc
import psutil
import os
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock

import pytest
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.core.service.resample.market_data_resampling_service import MarketDataResamplingService


def generate_ohlcv_data(count: int, symbol: str = "EURUSD") -> List[MarketDataModel]:
    """Generate realistic OHLCV test data quickly."""
    records = []
    base_price = 1.1000
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(count):
        # Simple price movement with small random changes
        price_change = (i % 100 - 50) * 0.0001
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


def create_mocked_service(test_data: List[MarketDataModel]) -> MarketDataResamplingService:
    """Create service with optimized mocks."""
    # Mock market data reader
    mock_reader = Mock()

    def get_chunked_data(symbol, timeframe, start_time, end_time, limit, offset):
        start_idx = offset
        end_idx = offset + limit
        chunk = test_data[start_idx:end_idx]
        return chunk

    mock_reader.get_symbol_data_range_paginated.side_effect = get_chunked_data

    # Mock message publisher
    mock_publisher = Mock()

    # Mock candle accumulator service with fast processing
    mock_accumulator = Mock()

    def mock_resample_to_timeframe(records, target_timeframe):
        # Fast simulation - just return a few candles
        reduction_factor = {
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.HOUR_1: 60,
            Timeframe.DAY_1: 1440
        }.get(target_timeframe, 1)

        num_candles = max(1, len(records) // reduction_factor)
        # Return lightweight mock data instead of full objects
        return list(range(num_candles))  # Just return integers instead of MarketDataModel objects

    mock_accumulator.resample_to_timeframe.side_effect = mock_resample_to_timeframe

    # Mock config
    mock_config = Mock()
    mock_config.pagination_limit = 50000  # Larger chunks for performance
    mock_config.historical_start_date = datetime(2024, 1, 1)

    return MarketDataResamplingService(
        market_data_reader=mock_reader,
        message_publisher=mock_publisher,
        candle_accumulator_service=mock_accumulator,
        resample_config=mock_config,
        state_persistence=None
    )


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestComprehensiveLoadPerformance:
    """Comprehensive load tests to find performance limits."""

    @pytest.mark.load
    def test_extreme_scaling_limits(self):
        """Test with increasing dataset sizes until we hit limits."""
        test_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
        results = []

        print("\n=== EXTREME SCALING TEST ===")
        print("Testing to find actual performance limits...")

        for size in test_sizes:
            print(f"\nTesting {size:,} records...")

            # Monitor memory before test
            gc.collect()
            memory_before = get_memory_usage()

            try:
                # Generate test data
                test_data = generate_ohlcv_data(size)
                memory_after_generation = get_memory_usage()

                service = create_mocked_service(test_data)

                # Time the processing
                start_time = time.time()
                response = service.resample_symbol_data_incremental(
                    symbol='EURUSD',
                    base_timeframe=Timeframe.MINUTE_1,
                    target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1]
                )
                end_time = time.time()

                processing_time = end_time - start_time
                records_per_sec = response.source_records_processed / processing_time
                memory_after = get_memory_usage()

                results.append({
                    'size': size,
                    'time': processing_time,
                    'rate': records_per_sec,
                    'processed': response.source_records_processed,
                    'memory_before': memory_before,
                    'memory_after_gen': memory_after_generation,
                    'memory_after': memory_after,
                    'memory_used': memory_after - memory_before
                })

                print(f"✅ SUCCESS: {response.source_records_processed:,} records in {processing_time:.2f}s")
                print(f"   Rate: {records_per_sec:,.0f} records/second")
                print(f"   Memory: {memory_before:.1f}MB → {memory_after:.1f}MB (+{memory_after - memory_before:.1f}MB)")

                # Cleanup
                del test_data, service, response
                gc.collect()

            except Exception as e:
                print(f"❌ FAILED at {size:,} records: {e}")
                break

        # Analyze results
        print("\n=== PERFORMANCE ANALYSIS ===")
        print("Size\t\tTime\t\tRate\t\tMemory\t\tProcessed")
        print("-" * 70)

        for result in results:
            print(f"{result['size']:,}\t\t{result['time']:.2f}s\t\t{result['rate']:,.0f}/sec\t\t{result['memory_used']:.1f}MB\t\t{result['processed']:,}")

        # Calculate scaling efficiency
        if len(results) >= 2:
            print("\n=== SCALING EFFICIENCY ===")
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]

                size_ratio = curr['size'] / prev['size']
                time_ratio = curr['time'] / prev['time']
                rate_ratio = curr['rate'] / prev['rate']
                complexity = time_ratio / size_ratio

                efficiency = rate_ratio * 100

                print(f"{prev['size']:,} → {curr['size']:,}: "
                      f"complexity={complexity:.2f}, "
                      f"efficiency={efficiency:.1f}% "
                      f"({'✅ GOOD' if complexity < 1.5 else '⚠️ DEGRADING'})")

        # Performance assessment
        if results:
            best_rate = max(r['rate'] for r in results)
            worst_rate = min(r['rate'] for r in results)
            degradation = (1 - worst_rate/best_rate) * 100

            print("\n=== FINAL ASSESSMENT ===")
            print(f"Best performance: {best_rate:,.0f} records/second")
            print(f"Worst performance: {worst_rate:,.0f} records/second")
            print(f"Performance degradation: {degradation:.1f}%")

            if degradation < 25:
                print("✅ EXCELLENT: Service scales well across all tested sizes")
            elif degradation < 50:
                print("⚠️ MODERATE: Some performance degradation at larger sizes")
            else:
                print("❌ POOR: Significant performance degradation")

    @pytest.mark.load
    def test_memory_pressure_breaking_point(self):
        """Test to find memory limits and breaking points."""
        print("\n=== MEMORY PRESSURE TEST ===")

        # Test increasingly large datasets until memory limits
        size = 1_000_000
        multiplier = 2

        while size <= 50_000_000:  # Cap at 50M records
            print(f"\nTesting {size:,} records for memory pressure...")

            memory_before = get_memory_usage()

            try:
                # Generate data and immediately test processing
                test_data = generate_ohlcv_data(size)
                memory_after_gen = get_memory_usage()

                # Check if data generation alone is causing issues
                if memory_after_gen > 8000:  # 8GB limit
                    print(f"❌ Memory limit reached during data generation: {memory_after_gen:.1f}MB")
                    break

                service = create_mocked_service(test_data)

                start_time = time.time()
                response = service.resample_symbol_data_incremental(
                    symbol='EURUSD',
                    base_timeframe=Timeframe.MINUTE_1,
                    target_timeframes=[Timeframe.MINUTE_5]  # Single timeframe for speed
                )
                end_time = time.time()

                processing_time = end_time - start_time
                records_per_sec = response.source_records_processed / processing_time
                memory_after = get_memory_usage()

                print(f"✅ SUCCESS: {response.source_records_processed:,} records")
                print(f"   Time: {processing_time:.2f}s ({records_per_sec:,.0f} rec/sec)")
                print(f"   Memory: {memory_before:.1f}MB → {memory_after:.1f}MB")

                # Cleanup before next iteration
                del test_data, service, response
                gc.collect()

                size *= multiplier

            except MemoryError:
                print(f"❌ MEMORY ERROR at {size:,} records")
                break
            except Exception as e:
                print(f"❌ ERROR at {size:,} records: {e}")
                break

        print("\n=== MEMORY LIMITS IDENTIFIED ===")
        print(f"Maximum testable dataset: {size // multiplier:,} records")
        print("Recommendation: Use chunked processing for larger datasets")
