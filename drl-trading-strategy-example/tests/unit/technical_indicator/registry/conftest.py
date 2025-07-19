"""
Shared test fixtures for drl-trading-strategy-example tests.

This module provides common fixtures and test utilities for all test modules
in the strategy example project.
"""

from typing import Optional, Type

import pytest
from drl_trading_common.base.base_indicator import BaseIndicator
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum
from drl_trading_strategy_example.technical_indicator.registry.indicator_class_registry import (
    IndicatorClassRegistry,
)
from pandas import DataFrame


class MockRsiIndicator(BaseIndicator):
    """Mock RSI indicator for testing purposes."""

    def __init__(self, length: int = 14) -> None:
        self.length = length

    def add(self, value: DataFrame) -> None:
        """Add new value to indicator (no-op for testing)."""
        pass

    def get_all(self) -> Optional[DataFrame]:
        """Get all computed values for testing."""
        return DataFrame({"rsi": [50.0, 60.0, 45.0]})

    def get_latest(self) -> Optional[DataFrame]:
        """Get latest computed value for testing."""
        return DataFrame({"rsi": [45.0]})


class MockAlternativeRsiIndicator(BaseIndicator):
    """Alternative mock RSI indicator for testing race conditions."""

    def __init__(self, length: int = 21) -> None:
        self.length = length

    def add(self, value: DataFrame) -> None:
        """Add new value to indicator (no-op for testing)."""
        pass

    def get_all(self) -> Optional[DataFrame]:
        """Get all computed values for testing."""
        return DataFrame({"rsi_alt": [55.0, 65.0, 40.0]})

    def get_latest(self) -> Optional[DataFrame]:
        """Get latest computed value for testing."""
        return DataFrame({"rsi_alt": [40.0]})


class MockMacdIndicator(BaseIndicator):
    """Mock MACD indicator for testing purposes."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def add(self, value: DataFrame) -> None:
        """Add new value to indicator (no-op for testing)."""
        pass

    def get_all(self) -> Optional[DataFrame]:
        """Get all computed values for testing."""
        return DataFrame({
            "macd": [0.5, 0.7, 0.3],
            "signal": [0.4, 0.6, 0.4],
            "histogram": [0.1, 0.1, -0.1]
        })

    def get_latest(self) -> Optional[DataFrame]:
        """Get latest computed value for testing."""
        return DataFrame({
            "macd": [0.3],
            "signal": [0.4],
            "histogram": [-0.1]
        })


@pytest.fixture
def registry() -> IndicatorClassRegistry:
    """Create a fresh indicator class registry instance for each test."""
    return IndicatorClassRegistry()


@pytest.fixture
def mock_rsi_indicator_class() -> Type[BaseIndicator]:
    """Provide MockRsiIndicator class for testing."""
    return MockRsiIndicator


@pytest.fixture
def mock_alternative_rsi_indicator_class() -> Type[BaseIndicator]:
    """Provide MockAlternativeRsiIndicator class for testing."""
    return MockAlternativeRsiIndicator


@pytest.fixture
def mock_macd_indicator_class() -> Type[BaseIndicator]:
    """Provide MockMacdIndicator class for testing."""
    return MockMacdIndicator


@pytest.fixture
def populated_registry(registry: IndicatorClassRegistry) -> IndicatorClassRegistry:
    """Provide a registry pre-populated with test indicator classes."""
    registry.register_indicator_class(IndicatorTypeEnum.RSI, MockRsiIndicator)
    return registry


@pytest.fixture
def indicator_type_rsi() -> IndicatorTypeEnum:
    """Provide RSI indicator type enum for testing."""
    return IndicatorTypeEnum.RSI
