"""
Unit tests for thread-safe TaLippIndicatorService.

This module tests the thread-safety guarantees of the enhanced service
implementation under concurrent access scenarios.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pytest
from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry import (
    IndicatorClassRegistry,
)
from drl_trading_strategy_example.technical_indicator.talipp_indicator_service import (
    TaLippIndicatorService,
)
from pandas import DataFrame


class MockThreadSafeIndicator(BaseIndicator):
    """Mock indicator with thread-safe operations for testing."""

    def __init__(self, length: int = 14) -> None:
        self.length = length
        self._lock = threading.RLock()
        self._data = []
        # Initialize with some default data for testing
        self._initialized = True

    def add(self, value: DataFrame) -> None:
        with self._lock:
            self._data.append(value)

    def get_all(self) -> Optional[DataFrame]:
        with self._lock:
            # Always return some data for testing purposes
            # In production, this would compute actual indicator values
            return DataFrame({"value": [50.0, 60.0, 45.0]})

    def get_latest(self) -> Optional[DataFrame]:
        with self._lock:
            # Always return some data for testing purposes
            # In production, this would return the latest computed value
            return DataFrame({"value": [45.0]})


class TestThreadSafeTaLippIndicatorService:
    """Test cases for thread-safe TaLippIndicatorService functionality."""

    @pytest.fixture
    def mock_registry(self) -> IndicatorClassRegistry:
        """Create a mock registry with test indicators."""
        registry = IndicatorClassRegistry()
        registry.register_indicator_class(IndicatorTypeEnum.RSI, MockThreadSafeIndicator)
        return registry

    @pytest.fixture
    def service(self, mock_registry: IndicatorClassRegistry) -> TaLippIndicatorService:
        """Create a service instance for each test."""
        return TaLippIndicatorService(mock_registry)

    def test_concurrent_registration_different_names(self, service: TaLippIndicatorService) -> None:
        """Test concurrent registration of indicators with different names."""
        # Given
        num_threads = 20
        results = []
        barrier = threading.Barrier(num_threads)

        def register_indicator(indicator_name: str) -> None:
            barrier.wait()
            try:
                service.register_instance(indicator_name, "rsi", length=14)
                results.append(f"success_{indicator_name}")
            except Exception as e:
                results.append(f"error_{indicator_name}_{type(e).__name__}")

        # When
        threads = []
        for i in range(num_threads):
            indicator_name = f"rsi_{i}"
            thread = threading.Thread(target=register_indicator, args=(indicator_name,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # All registrations should succeed since names are different
        successful_registrations = len([r for r in results if r.startswith("success")])
        assert successful_registrations == num_threads
        assert len(service) == num_threads

    def test_concurrent_registration_same_name(self, service: TaLippIndicatorService) -> None:
        """Test concurrent registration of indicators with the same name."""
        # Given
        indicator_name = "test_rsi"
        num_threads = 10
        results = []
        barrier = threading.Barrier(num_threads)

        def register_indicator() -> None:
            barrier.wait()
            try:
                service.register_instance(indicator_name, "rsi", length=14)
                results.append("success")
            except ValueError:
                results.append("error_ValueError")
            except Exception as e:
                results.append(f"error_{type(e).__name__}")

        # When
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=register_indicator)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # Only one registration should succeed, others should fail with ValueError
        successful_registrations = len([r for r in results if r == "success"])
        value_errors = len([r for r in results if r == "error_ValueError"])

        assert successful_registrations == 1
        assert value_errors == num_threads - 1
        assert len(service) == 1
        assert service.is_registered(indicator_name)

    def test_concurrent_read_operations(self, service: TaLippIndicatorService) -> None:
        """Test concurrent read operations are thread-safe."""
        # Given
        indicator_names = [f"rsi_{i}" for i in range(5)]
        for name in indicator_names:
            service.register_instance(name, "rsi", length=14)

        num_threads = 50
        read_results = []
        barrier = threading.Barrier(num_threads)

        def read_indicators() -> None:
            barrier.wait()
            try:
                # Perform multiple read operations
                result = {
                    "is_registered": service.is_registered(indicator_names[0]),
                    "get_all": service.get_all(indicator_names[1]) is not None,
                    "get_latest": service.get_latest(indicator_names[2]) is not None,
                    "names_count": len(service.get_registered_names()),
                    "total_count": len(service)
                }
                read_results.append(result)
            except Exception as e:
                read_results.append(f"error_{type(e).__name__}")

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
            assert result["is_registered"] is True
            assert result["get_all"] is True
            assert result["get_latest"] is True
            assert result["names_count"] == len(indicator_names)
            assert result["total_count"] == len(indicator_names)

    def test_concurrent_add_operations(self, service: TaLippIndicatorService) -> None:
        """Test concurrent add operations on the same indicator."""
        # Given
        indicator_name = "shared_rsi"
        service.register_instance(indicator_name, IndicatorTypeEnum.RSI, length=14)

        num_threads = 20
        operations_per_thread = 10
        results = []

        def add_data(thread_id: int) -> None:
            try:
                for i in range(operations_per_thread):
                    data = DataFrame({"price": [100.0 + thread_id + i]})
                    service.add(indicator_name, data)

                    # Occasionally read data too
                    if i % 3 == 0:
                        latest = service.get_latest(indicator_name)
                        assert latest is not None

                results.append(f"success_{thread_id}")
            except Exception as e:
                results.append(f"error_{thread_id}_{type(e).__name__}")

        # When
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=add_data, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # All operations should succeed
        successful_ops = len([r for r in results if r.startswith("success")])
        assert successful_ops == num_threads

    def test_concurrent_mixed_operations(self, service: TaLippIndicatorService) -> None:
        """Test mixed concurrent operations (register, read, add, unregister)."""
        # Given
        num_operations = 100
        results = []

        def perform_operation(operation_id: int) -> None:
            try:
                operation_type = operation_id % 5
                indicator_name = f"indicator_{operation_id % 10}"

                if operation_type == 0:
                    # Register operation
                    try:
                        service.register_instance(indicator_name, IndicatorTypeEnum.RSI, length=14)
                        results.append(f"register_success_{indicator_name}")
                    except ValueError:
                        results.append(f"register_exists_{indicator_name}")

                elif operation_type == 1:
                    # Read operation
                    is_registered = service.is_registered(indicator_name)
                    results.append(f"read_{indicator_name}_{is_registered}")

                elif operation_type == 2:
                    # Add operation
                    if service.is_registered(indicator_name):
                        data = DataFrame({"price": [100.0]})
                        service.add(indicator_name, data)
                        results.append(f"add_success_{indicator_name}")
                    else:
                        results.append(f"add_not_found_{indicator_name}")

                elif operation_type == 3:
                    # Get latest operation
                    if service.is_registered(indicator_name):
                        latest = service.get_latest(indicator_name)
                        results.append(f"latest_{indicator_name}_{latest is not None}")
                    else:
                        results.append(f"latest_not_found_{indicator_name}")

                else:
                    # Unregister operation
                    removed = service.unregister_instance(indicator_name)
                    results.append(f"unregister_{indicator_name}_{removed}")

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
        error_count = len([r for r in results if r.startswith("error")])
        error_percentage = (error_count / num_operations) * 100

        # Should have very few errors (< 1%)
        assert error_percentage < 1.0, f"Too many errors: {error_percentage}%"

    def test_concurrent_reset_operations(self, service: TaLippIndicatorService) -> None:
        """Test concurrent reset operations are safe."""
        # Given
        # Register some indicators first
        for i in range(5):
            service.register_instance(f"indicator_{i}", IndicatorTypeEnum.RSI, length=14)

        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def reset_and_check() -> tuple[int, int]:
            barrier.wait()
            # Get count before reset
            before_count = len(service)

            # Reset service
            service.reset()

            # Get count after reset
            after_count = len(service)

            return before_count, after_count

        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reset_and_check) for _ in range(num_threads)]
            [future.result() for future in as_completed(futures)]

        # Then
        # Service should be empty after all resets
        assert len(service) == 0
        assert len(service.get_registered_names()) == 0

    def test_stress_test_high_concurrency(self, service: TaLippIndicatorService) -> None:
        """Stress test with high concurrency load."""
        # Given
        num_threads = 50
        operations_per_thread = 30

        def stress_operations(thread_id: int) -> list[str]:
            results = []
            for i in range(operations_per_thread):
                try:
                    operation = (thread_id + i) % 6
                    indicator_name = f"stress_indicator_{thread_id % 5}"

                    if operation == 0:
                        # Register
                        try:
                            service.register_instance(indicator_name, IndicatorTypeEnum.RSI, length=14)
                            results.append("register_success")
                        except ValueError:
                            results.append("register_exists")
                    elif operation == 1:
                        # Check registration
                        exists = service.is_registered(indicator_name)
                        results.append(f"check_{exists}")
                    elif operation == 2:
                        # Add data
                        if service.is_registered(indicator_name):
                            data = DataFrame({"price": [100.0 + i]})
                            service.add(indicator_name, data)
                            results.append("add_success")
                        else:
                            results.append("add_not_found")
                    elif operation == 3:
                        # Get latest
                        if service.is_registered(indicator_name):
                            latest = service.get_latest(indicator_name)
                            results.append(f"latest_{latest is not None}")
                        else:
                            results.append("latest_not_found")
                    elif operation == 4:
                        # Get all names
                        names = service.get_registered_names()
                        results.append(f"names_{len(names)}")
                    else:
                        # Count indicators
                        count = len(service)
                        results.append(f"count_{count}")

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

        # Count error operations - should be minimal
        error_count = len([r for r in all_results if r.startswith("error")])
        error_percentage = (error_count / total_operations) * 100

        # Allow up to 1% errors due to timing issues
        assert error_percentage < 1.0, f"Too many errors: {error_percentage}%"

    def test_deadlock_prevention_nested_operations(self, service: TaLippIndicatorService) -> None:
        """Test that nested operations don't cause deadlocks."""
        # Given
        num_threads = 20
        timeout_seconds = 5

        def nested_operations(thread_id: int) -> str:
            try:
                indicator_name = f"nested_indicator_{thread_id % 3}"

                # Perform nested operations that could cause deadlocks
                if not service.is_registered(indicator_name):
                    service.register_instance(indicator_name, IndicatorTypeEnum.RSI, length=14)

                if service.is_registered(indicator_name):
                    data = DataFrame({"price": [100.0]})
                    service.add(indicator_name, data)

                    latest = service.get_latest(indicator_name)
                    if latest is not None:
                        names = service.get_registered_names()
                        return f"success_{len(names)}"

                return "success_0"
            except Exception as e:
                return f"error_{type(e).__name__}"

        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(nested_operations, i) for i in range(num_threads)]

            # Use timeout to detect deadlocks
            results = []
            for future in as_completed(futures, timeout=timeout_seconds):
                results.append(future.result())

        # Then
        # All operations should complete without timeout
        assert len(results) == num_threads
        successful_ops = len([r for r in results if r.startswith("success")])
        assert successful_ops > 0  # At least some operations should succeed

    def test_error_handling_consistency(self, service: TaLippIndicatorService) -> None:
        """Test that error handling is consistent under concurrent access."""
        # Given
        num_threads = 20
        results = []
        barrier = threading.Barrier(num_threads)

        def test_error_cases() -> None:
            barrier.wait()
            local_results = []

            # Test KeyError for non-existent indicator
            try:
                service.get_all("non_existent")
                local_results.append("error_no_keyerror")
            except KeyError:
                local_results.append("keyerror_get_all")
            except Exception as e:
                local_results.append(f"unexpected_{type(e).__name__}")

            # Test ValueError for duplicate registration
            try:
                service.register_instance("test_indicator", IndicatorTypeEnum.RSI, length=14)
                local_results.append("register_success")
            except ValueError:
                local_results.append("register_duplicate")
            except Exception as e:
                local_results.append(f"unexpected_{type(e).__name__}")

            results.extend(local_results)

        # When
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=test_error_cases)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        # Should have consistent error handling
        keyerror_count = len([r for r in results if r == "keyerror_get_all"])
        register_success_count = len([r for r in results if r == "register_success"])
        register_duplicate_count = len([r for r in results if r == "register_duplicate"])
        unexpected_count = len([r for r in results if r.startswith("unexpected")])

        # All threads should get KeyError for non-existent indicator
        assert keyerror_count == num_threads

        # Only one thread should successfully register, others should get ValueError
        assert register_success_count == 1
        assert register_duplicate_count == num_threads - 1

        # Should have no unexpected errors
        assert unexpected_count == 0
