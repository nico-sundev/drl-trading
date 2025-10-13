"""Benchmark tests comparing feature deserialization vs recomputation.

This module benchmarks two approaches for feature persistence:
1. Serialize/deserialize indicator state using jsonpickle
2. Recompute features from scratch

Tests use pytest-benchmark and are marked with @pytest.mark.benchmark
to be disabled by default. Run with: pytest -m benchmark

Key scenarios tested:
- Cold start: Deserialize vs. compute from scratch (1M candles)
- Warm continuation: Deserialize + update vs. full recomputation
- Multiple indicators: RSI, MACD, Bollinger Bands

Performance considerations:
- Serialization stores full indicator state (input history + buffers)
- Deserialization is typically faster than recomputation for large datasets
- Trade-off: Disk I/O vs. CPU computation
"""

import json
from pathlib import Path
from typing import Any

import jsonpickle
import pandas as pd
import pytest
from talipp.indicators import BB, MACD, RSI


@pytest.mark.benchmark
class TestFeatureDeserializationVsRecomputation:
    """Benchmark suite comparing deserialization vs recomputation strategies.

    These benchmarks help answer:
    1. Is it worth persisting indicator state to disk?
    2. What's the performance difference for different dataset sizes?
    3. How does it scale with multiple indicators?
    """

    def test_rsi_cold_start_small_dataset(
        self,
        benchmark_data_small: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 10k candles: deserialize vs compute.

        This tests the baseline performance with a small dataset.
        Expected: Recomputation should be competitive or faster.
        """
        # Given
        close_prices = benchmark_data_small["close"].tolist()
        rsi_period = 14
        persistence_path = temp_benchmark_dir / "rsi_small.json"

        # Pre-compute and serialize for deserialization test
        rsi_precomputed = RSI(period=rsi_period, input_values=close_prices)
        persistence_path.write_text(
            jsonpickle.encode(rsi_precomputed, unpicklable=True)
        )

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> RSI:
            serialized = persistence_path.read_text()
            return jsonpickle.decode(serialized)

        result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_rsi_cold_start_small_dataset_recompute(
        self,
        benchmark_data_small: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 10k candles: recomputation approach.

        Compare this with test_rsi_cold_start_small_dataset to see
        the performance difference.
        """
        # Given
        close_prices = benchmark_data_small["close"].tolist()
        rsi_period = 14

        # When/Then - Benchmark recomputation
        def recompute_approach() -> RSI:
            return RSI(period=rsi_period, input_values=close_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_rsi_cold_start_medium_dataset(
        self,
        benchmark_data_medium: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 100k candles: deserialize vs compute.

        This tests performance with a medium-sized dataset that's more
        representative of actual trading scenarios (several months of hourly data).
        Expected: Deserialization starts showing benefits.
        """
        # Given
        close_prices = benchmark_data_medium["close"].tolist()
        rsi_period = 14
        persistence_path = temp_benchmark_dir / "rsi_medium.json"

        # Pre-compute and serialize
        rsi_precomputed = RSI(period=rsi_period, input_values=close_prices)
        persistence_path.write_text(
            jsonpickle.encode(rsi_precomputed, unpicklable=True)
        )

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> RSI:
            serialized = persistence_path.read_text()
            return jsonpickle.decode(serialized)

        result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_rsi_cold_start_medium_dataset_recompute(
        self,
        benchmark_data_medium: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 100k candles: recomputation approach."""
        # Given
        close_prices = benchmark_data_medium["close"].tolist()
        rsi_period = 14

        # When/Then - Benchmark recomputation
        def recompute_approach() -> RSI:
            return RSI(period=rsi_period, input_values=close_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_rsi_cold_start_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 1M candles: deserialize vs compute.

        This is the main test case - 1M candles as requested.
        This represents years of hourly data or months of minute data.
        Expected: Deserialization should be significantly faster.
        """
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        rsi_period = 14
        persistence_path = temp_benchmark_dir / "rsi_large.json"

        # Pre-compute and serialize
        rsi_precomputed = RSI(period=rsi_period, input_values=close_prices)
        persistence_path.write_text(
            jsonpickle.encode(rsi_precomputed, unpicklable=True)
        )

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> RSI:
            serialized = persistence_path.read_text()
            return jsonpickle.decode(serialized)

        result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_rsi_cold_start_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI cold start with 1M candles: recomputation approach."""
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        rsi_period = 14

        # When/Then - Benchmark recomputation
        def recompute_approach() -> RSI:
            return RSI(period=rsi_period, input_values=close_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_macd_cold_start_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark MACD cold start with 1M candles: deserialize approach.

        MACD involves multiple EMA calculations, making it more
        computationally expensive than RSI.
        """
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        macd_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        persistence_path = temp_benchmark_dir / "macd_large.json"

        # Pre-compute and serialize
        macd_precomputed = MACD(**macd_params, input_values=close_prices)
        persistence_path.write_text(
            jsonpickle.encode(macd_precomputed, unpicklable=True)
        )

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> MACD:
            serialized = persistence_path.read_text()
            return jsonpickle.decode(serialized)

        result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_macd_cold_start_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark MACD cold start with 1M candles: recomputation approach."""
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        macd_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

        # When/Then - Benchmark recomputation
        def recompute_approach() -> MACD:
            return MACD(**macd_params, input_values=close_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_bb_cold_start_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark Bollinger Bands cold start with 1M candles: deserialize approach.

        BB requires SMA and standard deviation calculations.
        """
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        bb_params = {"period": 20, "std_dev": 2}
        persistence_path = temp_benchmark_dir / "bb_large.json"

        # Pre-compute and serialize
        bb_precomputed = BB(**bb_params, input_values=close_prices)
        persistence_path.write_text(
            jsonpickle.encode(bb_precomputed, unpicklable=True)
        )

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> BB:
            serialized = persistence_path.read_text()
            return jsonpickle.decode(serialized)

        result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_bb_cold_start_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark Bollinger Bands cold start with 1M candles: recomputation approach."""
        # Given
        close_prices = benchmark_data_large["close"].tolist()
        bb_params = {"period": 20, "std_dev": 2}

        # When/Then - Benchmark recomputation
        def recompute_approach() -> BB:
            return BB(**bb_params, input_values=close_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(close_prices)
        assert result[-1] is not None

    def test_all_indicators_cold_start_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark all indicators (RSI, MACD, BB) with 1M candles: deserialize approach.

        This tests the combined performance of multiple indicators,
        which is closer to real-world usage.
        """
        # Given
        close_prices = benchmark_data_large["close"].tolist()

        # Pre-compute and serialize all indicators
        rsi = RSI(period=14, input_values=close_prices)
        macd = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=close_prices)
        bb = BB(period=20, std_dev=2, input_values=close_prices)

        indicators_state = {
            "rsi": jsonpickle.encode(rsi, unpicklable=True),
            "macd": jsonpickle.encode(macd, unpicklable=True),
            "bb": jsonpickle.encode(bb, unpicklable=True),
        }

        persistence_path = temp_benchmark_dir / "all_indicators_large.json"
        persistence_path.write_text(json.dumps(indicators_state))

        # When/Then - Benchmark deserialization
        def deserialize_approach() -> tuple[RSI, MACD, BB]:
            state = json.loads(persistence_path.read_text())
            return (
                jsonpickle.decode(state["rsi"]),
                jsonpickle.decode(state["macd"]),
                jsonpickle.decode(state["bb"]),
            )

        rsi_result, macd_result, bb_result = benchmark(deserialize_approach)

        # Verify correctness
        assert len(rsi_result) == len(close_prices)
        assert len(macd_result) == len(close_prices)
        assert len(bb_result) == len(close_prices)

    def test_all_indicators_cold_start_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark all indicators (RSI, MACD, BB) with 1M candles: recomputation approach."""
        # Given
        close_prices = benchmark_data_large["close"].tolist()

        # When/Then - Benchmark recomputation
        def recompute_approach() -> tuple[RSI, MACD, BB]:
            rsi = RSI(period=14, input_values=close_prices)
            macd = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=close_prices)
            bb = BB(period=20, std_dev=2, input_values=close_prices)
            return rsi, macd, bb

        rsi_result, macd_result, bb_result = benchmark(recompute_approach)

        # Verify correctness
        assert len(rsi_result) == len(close_prices)
        assert len(macd_result) == len(close_prices)
        assert len(bb_result) == len(close_prices)


@pytest.mark.benchmark
class TestFeatureWarmContinuation:
    """Benchmark suite for warm continuation scenarios.

    This tests the scenario where we:
    1. Load persisted indicator state
    2. Add new candles incrementally

    This is closer to real-time/production usage where features
    are computed incrementally as new data arrives.
    """

    def test_rsi_warm_continuation_medium_dataset(
        self,
        benchmark_data_medium: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI warm continuation with 100k candles: deserialize + update 100 new candles.

        This tests the realistic scenario where we load historical state
        and update with new data.
        """
        # Given - Split data: 99,900 historical + 100 new
        historical_prices = benchmark_data_medium["close"].iloc[:-100].tolist()
        new_prices = benchmark_data_medium["close"].iloc[-100:].tolist()
        rsi_period = 14
        persistence_path = temp_benchmark_dir / "rsi_warm_medium.json"

        # Pre-compute and serialize historical state
        rsi_historical = RSI(period=rsi_period, input_values=historical_prices)
        persistence_path.write_text(
            jsonpickle.encode(rsi_historical, unpicklable=True)
        )

        # When/Then - Benchmark deserialize + update
        def deserialize_and_update_approach() -> RSI:
            serialized = persistence_path.read_text()
            rsi = jsonpickle.decode(serialized)
            # Add new prices incrementally
            for price in new_prices:
                rsi.add(price)
            return rsi

        result = benchmark(deserialize_and_update_approach)

        # Verify correctness
        assert len(result) == len(historical_prices) + len(new_prices)
        assert result[-1] is not None

    def test_rsi_warm_continuation_medium_dataset_recompute(
        self,
        benchmark_data_medium: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI warm continuation with 100k candles: full recomputation.

        Compare with deserialize+update to see the benefit of incremental updates.
        """
        # Given
        all_prices = benchmark_data_medium["close"].tolist()
        rsi_period = 14

        # When/Then - Benchmark full recomputation
        def recompute_approach() -> RSI:
            return RSI(period=rsi_period, input_values=all_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(all_prices)
        assert result[-1] is not None

    def test_rsi_warm_continuation_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI warm continuation: deserialize + update 100 new candles.

        Scenario: Load persisted state and update with new data.
        This is the most common production scenario.
        """
        # Given - Split data: 999,900 historical + 100 new
        historical_prices = benchmark_data_large["close"].iloc[:-100].tolist()
        new_prices = benchmark_data_large["close"].iloc[-100:].tolist()
        rsi_period = 14
        persistence_path = temp_benchmark_dir / "rsi_warm.json"

        # Pre-compute and serialize historical state
        rsi_historical = RSI(period=rsi_period, input_values=historical_prices)
        persistence_path.write_text(
            jsonpickle.encode(rsi_historical, unpicklable=True)
        )

        # When/Then - Benchmark deserialize + update
        def deserialize_and_update_approach() -> RSI:
            serialized = persistence_path.read_text()
            rsi = jsonpickle.decode(serialized)
            # Add new prices incrementally
            for price in new_prices:
                rsi.add(price)
            return rsi

        result = benchmark(deserialize_and_update_approach)

        # Verify correctness
        assert len(result) == len(historical_prices) + len(new_prices)
        assert result[-1] is not None

    def test_rsi_warm_continuation_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark RSI warm continuation: full recomputation with all data.

        Compare with deserialize+update to see the benefit of incremental updates.
        """
        # Given
        all_prices = benchmark_data_large["close"].tolist()
        rsi_period = 14

        # When/Then - Benchmark full recomputation
        def recompute_approach() -> RSI:
            return RSI(period=rsi_period, input_values=all_prices)

        result = benchmark(recompute_approach)

        # Verify correctness
        assert len(result) == len(all_prices)
        assert result[-1] is not None

    def test_all_indicators_warm_continuation_large_dataset(
        self,
        benchmark_data_large: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Benchmark all indicators warm continuation: deserialize + update.

        This tests the combined performance of multiple indicators
        in a warm continuation scenario.
        """
        # Given - Split data
        historical_prices = benchmark_data_large["close"].iloc[:-100].tolist()
        new_prices = benchmark_data_large["close"].iloc[-100:].tolist()

        # Pre-compute and serialize all indicators
        rsi = RSI(period=14, input_values=historical_prices)
        macd = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=historical_prices)
        bb = BB(period=20, std_dev=2, input_values=historical_prices)

        indicators_state = {
            "rsi": jsonpickle.encode(rsi, unpicklable=True),
            "macd": jsonpickle.encode(macd, unpicklable=True),
            "bb": jsonpickle.encode(bb, unpicklable=True),
        }

        persistence_path = temp_benchmark_dir / "all_indicators_warm.json"
        persistence_path.write_text(json.dumps(indicators_state))

        # When/Then - Benchmark deserialize + update
        def deserialize_and_update_approach() -> tuple[RSI, MACD, BB]:
            state = json.loads(persistence_path.read_text())
            rsi = jsonpickle.decode(state["rsi"])
            macd = jsonpickle.decode(state["macd"])
            bb = jsonpickle.decode(state["bb"])

            # Add new prices incrementally to all indicators
            for price in new_prices:
                rsi.add(price)
                macd.add(price)
                bb.add(price)

            return rsi, macd, bb

        rsi_result, macd_result, bb_result = benchmark(deserialize_and_update_approach)

        # Verify correctness
        expected_length = len(historical_prices) + len(new_prices)
        assert len(rsi_result) == expected_length
        assert len(macd_result) == expected_length
        assert len(bb_result) == expected_length

    def test_all_indicators_warm_continuation_large_dataset_recompute(
        self,
        benchmark_data_large: pd.DataFrame,
        benchmark: Any,
    ) -> None:
        """Benchmark all indicators warm continuation: full recomputation."""
        # Given
        all_prices = benchmark_data_large["close"].tolist()

        # When/Then - Benchmark full recomputation
        def recompute_approach() -> tuple[RSI, MACD, BB]:
            rsi = RSI(period=14, input_values=all_prices)
            macd = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=all_prices)
            bb = BB(period=20, std_dev=2, input_values=all_prices)
            return rsi, macd, bb

        rsi_result, macd_result, bb_result = benchmark(recompute_approach)

        # Verify correctness
        assert len(rsi_result) == len(all_prices)
        assert len(macd_result) == len(all_prices)
        assert len(bb_result) == len(all_prices)
