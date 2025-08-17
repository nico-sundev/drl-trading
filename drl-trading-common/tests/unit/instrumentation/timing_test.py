"""Tests for generic timing utilities."""
from __future__ import annotations

import time
from typing import List, Dict, Any

from drl_trading_common.instrumentation.timing import timing, time_block


def test_timing_decorator_basic() -> None:
    """Given a decorated function When it executes Then it emits timing metric via callback."""
    # Given
    metrics: List[Dict[str, Any]] = []

    @timing(name="sample_fn", emit=lambda m: metrics.append(m))
    def sample() -> str:
        time.sleep(0.01)
        return "ok"

    # When
    result = sample()

    # Then
    assert result == "ok"
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric["metric"] == "timing"
    assert metric["name"] == "sample_fn"
    assert metric["duration_ms"] >= 10


def test_time_block_context_manager() -> None:
    """Given a timing block When code executes Then a metric is emitted."""
    # Given
    metrics: List[Dict[str, Any]] = []

    # When
    with time_block("block", emit=lambda m: metrics.append(m)):
        time.sleep(0.005)

    # Then
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric["name"] == "block"
    assert metric["duration_ms"] >= 5
