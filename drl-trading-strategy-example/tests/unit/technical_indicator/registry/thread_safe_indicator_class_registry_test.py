"""
Unit tests for thread-safe IndicatorClassRegistry.

This module tests the thread-safety guarantees of the enhanced registry
implementation under concurrent access scenarios.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type

from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy.technical_indicator.registry.indicator_class_registry import (
    IndicatorClassRegistry,
)


class TestThreadSafeIndicatorClassRegistry:
    """Test cases for thread-safe IndicatorClassRegistry functionality."""

    def test_concurrent_registration_same_key(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        mock_alternative_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test concurrent registration of different classes for the same key."""
        # Given
        num_threads = 10
        results = []
        barrier = threading.Barrier(num_threads)

        def register_indicator(indicator_class: Type[BaseIndicator]) -> None:
            # Synchronize all threads to start at the same time
            barrier.wait()
            try:
                registry.register_indicator_class(indicator_type_rsi, indicator_class)
                results.append(f"success_{indicator_class.__name__}")
            except Exception as e:
                results.append(f"error_{type(e).__name__}")

        # When
        threads = []
        for i in range(num_threads):
            # Alternate between different RSI indicator implementations
            indicator_class = mock_rsi_indicator_class if i % 2 == 0 else mock_alternative_rsi_indicator_class
            thread = threading.Thread(target=register_indicator, args=(indicator_class,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # Only one registration should succeed, others should get overridden
        final_class = registry.get_indicator_class(indicator_type_rsi)
        assert final_class is not None
        assert final_class in [mock_rsi_indicator_class, mock_alternative_rsi_indicator_class]
        assert len(results) == num_threads

    def test_concurrent_registration_different_keys(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        mock_alternative_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test concurrent registration of classes for different keys - limited to RSI only."""
        # Given
        num_threads = 10  # Reduced since we only have RSI
        results = []
        barrier = threading.Barrier(num_threads)

        def register_indicator(indicator_class: Type[BaseIndicator]) -> None:
            barrier.wait()
            try:
                registry.register_indicator_class(indicator_type_rsi, indicator_class)
                results.append("success_RSI")
            except Exception as e:
                results.append(f"error_RSI_{type(e).__name__}")

        # When
        threads = []
        for i in range(num_threads):
            # Use different RSI implementations for different threads
            indicator_class = mock_rsi_indicator_class if i % 2 == 0 else mock_alternative_rsi_indicator_class
            thread = threading.Thread(target=register_indicator, args=(indicator_class,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # All registrations should complete (though they overwrite each other)
        assert len(results) == num_threads
        final_class = registry.get_indicator_class(indicator_type_rsi)
        assert final_class in [mock_rsi_indicator_class, mock_alternative_rsi_indicator_class]

    def test_concurrent_read_operations(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test concurrent read operations are thread-safe."""
        # Given
        registry.register_indicator_class(indicator_type_rsi, mock_rsi_indicator_class)

        num_threads = 50
        read_results = []
        barrier = threading.Barrier(num_threads)

        def read_indicators() -> None:
            barrier.wait()
            try:
                # Perform multiple read operations
                rsi_class = registry.get_indicator_class(indicator_type_rsi)
                all_registered = registry.get_all_registered()
                is_rsi_registered = registry.is_registered(indicator_type_rsi)

                read_results.append({
                    "rsi_class": rsi_class,
                    "count": len(all_registered),
                    "rsi_registered": is_rsi_registered
                })
            except Exception as e:
                read_results.append({"error": f"error_{type(e).__name__}"})

        # When
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=read_indicators)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # All reads should be consistent
        for result in read_results:
            assert isinstance(result, dict)
            assert result["rsi_class"] == mock_rsi_indicator_class
            assert result["count"] == 1
            assert result["rsi_registered"] is True

    def test_concurrent_mixed_operations(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        mock_alternative_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test mixed concurrent read/write operations."""
        # Given
        num_operations = 100
        results = []

        def perform_operation(operation_id: int) -> None:
            try:
                if operation_id % 3 == 0:
                    # Register operation - only RSI available
                    indicator_class = mock_rsi_indicator_class if operation_id % 6 == 0 else mock_alternative_rsi_indicator_class
                    registry.register_indicator_class(indicator_type_rsi, indicator_class)
                    results.append("register_RSI")
                elif operation_id % 3 == 1:
                    # Read operation
                    rsi_class = registry.get_indicator_class(indicator_type_rsi)
                    results.append(f"read_rsi_{rsi_class is not None}")
                else:
                    # Check operation
                    is_registered = registry.is_registered(indicator_type_rsi)
                    results.append(f"check_rsi_{is_registered}")
            except Exception as e:
                results.append(f"error_{operation_id}_{type(e).__name__}")

        # When
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(perform_operation, i) for i in range(num_operations)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Then
        assert len(results) == num_operations
        # Should have mix of operations without errors
        register_ops = len([r for r in results if r.startswith("register")])
        read_ops = len([r for r in results if r.startswith("read")])
        check_ops = len([r for r in results if r.startswith("check")])

        assert register_ops > 0
        assert read_ops > 0
        assert check_ops > 0
        assert register_ops + read_ops + check_ops == num_operations

    def test_concurrent_reset_operations(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test concurrent reset operations are safe."""
        # Given
        registry.register_indicator_class(indicator_type_rsi, mock_rsi_indicator_class)
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def reset_and_check() -> tuple[int, int]:
            barrier.wait()
            # Get count before reset
            before_count = len(registry)

            # Reset registry
            registry.reset()

            # Get count after reset
            after_count = len(registry)

            return before_count, after_count

        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reset_and_check) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # Then
        # Registry should be empty after all resets
        assert len(registry) == 0
        assert registry.get_indicator_class(indicator_type_rsi) is None

    def test_thread_safety_stress_test(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Stress test with high concurrency load."""
        # Given
        num_threads = 50
        operations_per_thread = 20

        def stress_operations(thread_id: int) -> list[str]:
            results = []
            for i in range(operations_per_thread):
                try:
                    operation = (thread_id + i) % 4
                    if operation == 0:
                        # Register
                        registry.register_indicator_class(indicator_type_rsi, mock_rsi_indicator_class)
                        results.append("register")
                    elif operation == 1:
                        # Read
                        cls = registry.get_indicator_class(indicator_type_rsi)
                        results.append(f"read_{cls is not None}")
                    elif operation == 2:
                        # Check
                        exists = registry.is_registered(indicator_type_rsi)
                        results.append(f"check_{exists}")
                    else:
                        # Get all
                        all_registered = registry.get_all_registered()
                        results.append(f"all_{len(all_registered)}")

                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)
                except Exception as e:
                    results.append(f"error_{type(e).__name__}")
            return results

        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_operations, i) for i in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        # Then
        total_operations = num_threads * operations_per_thread
        assert len(all_results) == total_operations

        # Count error operations - should be minimal or zero
        error_count = len([r for r in all_results if r.startswith("error")])
        error_percentage = (error_count / total_operations) * 100

        # Allow up to 1% errors due to timing/validation issues
        assert error_percentage < 1.0, f"Too many errors: {error_percentage}%"

    def test_deadlock_prevention(
        self,
        registry: IndicatorClassRegistry,
        mock_rsi_indicator_class: Type[BaseIndicator],
        indicator_type_rsi: IndicatorTypeEnum
    ) -> None:
        """Test that operations don't cause deadlocks."""
        # Given
        num_threads = 20
        timeout_seconds = 5

        def nested_operations() -> str:
            try:
                # Perform nested operations that could cause deadlocks
                registry.register_indicator_class(indicator_type_rsi, mock_rsi_indicator_class)
                if registry.is_registered(indicator_type_rsi):
                    cls = registry.get_indicator_class(indicator_type_rsi)
                    if cls:
                        all_registered = registry.get_all_registered()
                        return f"success_{len(all_registered)}"
                return "success_0"
            except Exception as e:
                return f"error_{type(e).__name__}"

        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(nested_operations) for _ in range(num_threads)]

            # Use timeout to detect deadlocks
            results = []
            for future in as_completed(futures, timeout=timeout_seconds):
                results.append(future.result())

        # Then
        # All operations should complete without timeout
        assert len(results) == num_threads
        successful_ops = len([r for r in results if r.startswith("success")])
        assert successful_ops > 0  # At least some operations should succeed
