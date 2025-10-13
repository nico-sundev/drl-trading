"""Shared fixtures for benchmark tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Generator
import tempfile


@pytest.fixture(scope="session")
def benchmark_data_small() -> pd.DataFrame:
    """Generate 10k OHLCV candles for quick benchmarks.

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    return _generate_ohlcv_data(num_candles=10_000, seed=42)


@pytest.fixture(scope="session")
def benchmark_data_medium() -> pd.DataFrame:
    """Generate 100k OHLCV candles for medium-scale benchmarks.

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    return _generate_ohlcv_data(num_candles=100_000, seed=42)


@pytest.fixture(scope="session")
def benchmark_data_large() -> pd.DataFrame:
    """Generate 1M OHLCV candles for large-scale benchmarks.

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    return _generate_ohlcv_data(num_candles=1_000_000, seed=42)


@pytest.fixture(scope="function")
def temp_benchmark_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for benchmark artifacts.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory(prefix="benchmark_") as tmpdir:
        yield Path(tmpdir)


def _generate_ohlcv_data(num_candles: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV data for benchmarking.

    Creates synthetic price data with:
    - Realistic price movements (random walk with drift)
    - Proper OHLC relationships (H >= C/O >= L)
    - Volume variation

    Args:
        num_candles: Number of candles to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(seed)

    # Generate close prices using geometric brownian motion
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, num_candles)  # ~1bp mean, 200bp std
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = []
    prev_close = base_price

    for i, close in enumerate(close_prices):
        # Open is previous close (realistic for continuous markets)
        open_price = prev_close

        # High/Low with realistic spread (0.1% - 0.5%)
        spread = abs(np.random.normal(0.002, 0.001))
        high = max(open_price, close) * (1 + spread)
        low = min(open_price, close) * (1 - spread)

        # Volume with some correlation to price movement
        base_volume = 10_000
        volatility_factor = abs(close - open_price) / open_price
        volume = int(base_volume * (1 + np.random.exponential(volatility_factor * 100)))

        data.append({
            "open": round(open_price, 5),
            "high": round(high, 5),
            "low": round(low, 5),
            "close": round(close, 5),
            "volume": volume
        })

        prev_close = close

    return pd.DataFrame(data)
