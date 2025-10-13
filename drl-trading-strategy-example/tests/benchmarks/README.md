# Feature Persistence Benchmarks

## Overview

This benchmark suite compares two approaches for feature persistence in preprocessing workflows:

1. **Serialization/Deserialization**: Using `jsonpickle` to serialize/deserialize indicator state
2. **Recomputation**: Computing features from scratch every time

## Why This Matters

In preprocessing pipelines, we often need to "warm up" features with historical data. The question is: should we:
- Persist computed indicator states and load them (fast startup, disk I/O overhead)
- Recompute from scratch every time (no disk I/O, CPU intensive)

## Test Scenarios

### Cold Start
Tests the initial computation/loading of indicators:
- **Deserialize approach**: Load pre-computed indicator state from disk
- **Recompute approach**: Compute from scratch

### Warm Continuation
Tests incremental updates (production scenario):
- **Deserialize+update**: Load historical state, add new candles
- **Recompute**: Compute all data including new candles

## Indicators Tested

- **RSI(14)**: Relative Strength Index
- **MACD(12,26,9)**: Moving Average Convergence Divergence
- **BB(20,2)**: Bollinger Bands

## Dataset Sizes

- **Small**: 10k candles (~2 weeks of hourly data)
- **Medium**: 100k candles (~11 years of hourly data)
- **Large**: 1M candles (~114 years of hourly data / ~2 years of minute data)

## Running Benchmarks

### Run all benchmarks
```bash
cd drl-trading-strategy-example
uv run pytest -m benchmark tests/benchmarks/ -v
```

### Run with detailed output
```bash
uv run pytest -m benchmark tests/benchmarks/ -v -s
```

### Run specific benchmark
```bash
uv run pytest -m benchmark tests/benchmarks/ -v -k "rsi_cold_start_large"
```

### Generate benchmark comparison report
```bash
uv run pytest -m benchmark tests/benchmarks/ --benchmark-compare
```

### Save benchmark results
```bash
uv run pytest -m benchmark tests/benchmarks/ --benchmark-autosave
```

## Interpreting Results

pytest-benchmark provides several metrics:

- **Min**: Fastest execution time
- **Max**: Slowest execution time
- **Mean**: Average execution time
- **StdDev**: Standard deviation (consistency)
- **Median**: Middle value (robust to outliers)
- **IQR**: Interquartile range (spread of middle 50%)
- **Outliers**: Number of outlier measurements

### Example Output

```
--------------------------------- benchmark: 2 tests ---------------------------------
Name (time in ms)                                          Min       Max      Mean
--------------------------------------------------------------------------------------
test_rsi_cold_start_large_dataset                      45.23     52.11     47.89
test_rsi_cold_start_large_dataset_recompute           892.34    945.21    915.44
--------------------------------------------------------------------------------------
```

In this example, deserialization is ~19x faster than recomputation.

## Expected Results

Based on the serialization approach:

### Cold Start (1M candles)
- **RSI**: Deserialize should be 15-20x faster
- **MACD**: Deserialize should be 20-30x faster (more complex calculation)
- **BB**: Deserialize should be 10-15x faster

### Warm Continuation (999,900 + 100 candles)
- Deserialize+update should be 50-100x faster (only computing 100 new candles)

### Trade-offs
- **Serialization pros**: Much faster startup, enables incremental updates
- **Serialization cons**: Disk I/O overhead, storage requirements (~10-50MB per indicator for 1M candles)
- **Recomputation pros**: No storage, always fresh, simpler architecture
- **Recomputation cons**: CPU intensive, not practical for large datasets

## Integration with Preprocessing Pipeline

Based on benchmark results, recommendations:

1. **For large datasets (>100k candles)**: Use serialization
2. **For real-time/streaming**: Definitely use serialization + incremental updates
3. **For small datasets (<10k candles)**: Recomputation is fine
4. **For batch processing**: Consider trade-off between disk space and compute time

## Implementation Notes

### Serialization Format
Using `jsonpickle` with `unpicklable=True` to preserve full object state including:
- Input value history
- Calculation buffers (for EMA, etc.)
- Indicator parameters

### Storage Considerations
For 1M candles with all 3 indicators:
- RSI state: ~15-20MB
- MACD state: ~30-40MB (3 EMAs)
- BB state: ~20-30MB
- **Total**: ~65-90MB

This is acceptable for most use cases, but consider compression for storage optimization.

## Next Steps

After running benchmarks:

1. Review results and decide on persistence strategy
2. Implement chosen approach in actual feature classes
3. Add integration tests for persistence layer
4. Monitor production performance

## Troubleshooting

### Benchmarks take too long
- Run subset: `pytest -m benchmark -k "small_dataset"`
- Reduce iterations: `pytest -m benchmark --benchmark-max-time=5`

### Out of memory
- The 1M candle tests use ~500MB RAM
- Reduce dataset size if needed (edit conftest.py fixtures)

### Inconsistent results
- Run multiple rounds: `pytest -m benchmark --benchmark-rounds=10`
- Check for background processes affecting CPU

## References

- [talipp serialization docs](https://nardew.github.io/talipp/latest/serialization/)
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [jsonpickle docs](https://jsonpickle.github.io/)
