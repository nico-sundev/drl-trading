# Feature Persistence Benchmark Suite

## Quick Start

```bash
# Install dependencies
cd drl-trading-strategy-example
uv sync --group dev-full

# Run all benchmarks
uv run pytest -m benchmark tests/benchmarks/ -v

# Run specific scenario
uv run pytest -m benchmark -k "cold_start_medium" -v

# Generate HTML report
uv run pytest -m benchmark tests/benchmarks/ --benchmark-histogram
```

## What Was Built

A comprehensive benchmark suite comparing two approaches for feature persistence:

1. **Serialization**: Using jsonpickle to save/load indicator state
2. **Recomputation**: Computing features from scratch

### Test Coverage

- **18 benchmark tests** total
- **3 indicators**: RSI(14), MACD(12,26,9), Bollinger Bands(20,2)
- **3 dataset sizes**: 10k, 100k, 1M candles
- **2 scenarios**: Cold start, Warm continuation

### Key Files Created

```
tests/benchmarks/
├── __init__.py                              # Package marker
├── conftest.py                              # Shared fixtures (OHLCV data generators)
├── feature_persistence_benchmark_test.py    # 18 benchmark tests
├── README.md                                # Detailed documentation
└── RESULTS.md                               # Performance analysis & recommendations
```

## Results Summary

**Bottom line**: Recomputation is 2x faster than jsonpickle deserialization across all scenarios.

| Scenario | Recompute | Deserialize | Winner |
|----------|-----------|-------------|--------|
| Cold start 10k | 17ms | 36ms | Recompute ✅ |
| Cold start 100k | 168ms | 362ms | Recompute ✅ |
| Warm 100k (99.9k+100) | 174ms | 383ms | Recompute ✅ |

**Recommendation**: Stick with recomputation for current use case.

## Architecture Integration

### Current Preprocessing Workflow

```python
# Your warmup process (simplified)
historical_data = load_ohlcv(symbol, start, end)
indicator = RSI(period=14, input_values=historical_data["close"].tolist())

# New data arrives
new_candle = get_latest_candle(symbol)
indicator.add(new_candle["close"])
```

### What We Tested

```python
# Approach 1: Serialize/deserialize (tested, not recommended)
import jsonpickle

# Save state
serialized = jsonpickle.encode(indicator, unpicklable=True)
Path("rsi_state.json").write_text(serialized)

# Load state
indicator = jsonpickle.decode(Path("rsi_state.json").read_text())

# Approach 2: Recompute (tested, recommended)
# Just compute again - it's faster!
indicator = RSI(period=14, input_values=all_data)
```

## Why This Matters

In preprocessing, we often "warm up" features with historical data before processing new data. The question was:

> Should we persist computed indicator states to avoid recomputation?

**Answer**: No, recomputation is faster and simpler with current tools.

## Pytest Configuration

Benchmarks are **disabled by default** to keep normal test runs fast.

### pyproject.toml Updates

```toml
[tool.pytest.ini_options]
addopts = ["-ra", "-m not slow and not benchmark", "--strict-markers"]
markers = [
    "slow: marks tests as slow (deselected by default)",
    "benchmark: marks tests as benchmarks (deselected by default, run with -m benchmark)",
]
```

### Dependencies Added

```toml
dependencies = [
    # ... existing deps
    "jsonpickle>=3.0.0",  # Indicator state serialization
]

[dependency-groups]
test = [
    # ... existing test deps
    "pytest-benchmark>=4.0.0",  # For performance testing
]
```

## Usage Examples

### Run All Benchmarks

```bash
uv run pytest -m benchmark tests/benchmarks/ -v
```

### Run Cold Start Tests Only

```bash
uv run pytest -m benchmark -k "cold_start" -v
```

### Run Warm Continuation Tests Only

```bash
uv run pytest -m benchmark -k "warm_continuation" -v
```

### Run Large Dataset Tests Only (1M candles)

```bash
uv run pytest -m benchmark -k "large_dataset" -v
```

### Compare Specific Indicators

```bash
# RSI only
uv run pytest -m benchmark -k "rsi" -v

# MACD only
uv run pytest -m benchmark -k "macd" -v

# All indicators together
uv run pytest -m benchmark -k "all_indicators" -v
```

### Save Benchmark History

```bash
# First run
uv run pytest -m benchmark tests/benchmarks/ --benchmark-autosave

# Make code changes...

# Second run (will compare automatically)
uv run pytest -m benchmark tests/benchmarks/ --benchmark-autosave --benchmark-compare
```

## What You Can Learn

### From the Tests

1. **Performance characteristics** of talipp indicators at scale
2. **Serialization overhead** with jsonpickle
3. **Incremental update costs** vs. full recomputation
4. **Memory usage patterns** for different dataset sizes

### From the Implementation

1. **Pytest benchmark patterns** for real-world performance testing
2. **Fixture design** for generating realistic test data
3. **pytest.mark** usage for conditional test execution
4. **Test organization** for benchmark suites

## Future Extensions

If you want to explore further:

### 1. Test Binary Formats

```python
# In conftest.py, add pickle-based fixtures
import pickle

@pytest.fixture
def pickle_serialized_rsi(benchmark_data_large):
    rsi = RSI(period=14, input_values=benchmark_data_large["close"].tolist())
    return pickle.dumps(rsi)
```

### 2. Test More Indicators

```python
# Add StochRSI, ATR, ADX tests
from talipp.indicators import StochRSI, ATR, ADX
```

### 3. Test Multi-Timeframe

```python
# Generate data for multiple timeframes
@pytest.fixture
def multi_timeframe_data():
    return {
        "1m": generate_ohlcv(60000),
        "5m": generate_ohlcv(12000),
        "1h": generate_ohlcv(1000),
    }
```

### 4. Profile Memory Usage

```bash
# Add memory profiling
uv add memory-profiler

# Run with memory tracking
uv run pytest -m benchmark --benchmark-max-time=5 --memprof
```

## Troubleshooting

### Tests Run Too Long

```bash
# Limit execution time
uv run pytest -m benchmark --benchmark-max-time=5

# Or run subset
uv run pytest -m benchmark -k "small_dataset"
```

### Out of Memory (1M candle tests)

The large dataset tests use ~500MB RAM. If you hit memory limits:

1. Reduce dataset size in `conftest.py`:
```python
@pytest.fixture(scope="session")
def benchmark_data_large() -> pd.DataFrame:
    return _generate_ohlcv_data(num_candles=500_000, seed=42)  # Reduced
```

2. Or skip large tests:
```bash
uv run pytest -m benchmark -k "not large_dataset"
```

### Inconsistent Results

Run multiple rounds for statistical significance:

```bash
uv run pytest -m benchmark --benchmark-rounds=10 --benchmark-warmup=on
```

## Development Guidelines

### Adding New Benchmarks

Follow the established pattern:

```python
@pytest.mark.benchmark
class TestMyNewScenario:
    """Docstring explaining what this tests."""

    def test_my_approach_deserialize(
        self,
        benchmark_data_medium: pd.DataFrame,
        temp_benchmark_dir: Path,
        benchmark: Any,
    ) -> None:
        """Test description."""
        # Given
        data = benchmark_data_medium["close"].tolist()

        # Setup (not benchmarked)
        # ... prepare state

        # When/Then - Benchmark
        def approach():
            # ... your code here
            return result

        result = benchmark(approach)

        # Verify correctness
        assert result is not None
```

### Benchmark Best Practices

1. **Separate setup from benchmark**: Only benchmark the target operation
2. **Verify correctness**: Always assert results are valid
3. **Use realistic data**: Leverage existing fixtures
4. **Document expectations**: Explain what should be faster/slower
5. **Keep it simple**: One operation per test

## References

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [talipp serialization](https://nardew.github.io/talipp/latest/serialization/)
- [jsonpickle docs](https://jsonpickle.github.io/)
- [RESULTS.md](./RESULTS.md) - Detailed performance analysis

---

**Questions?** Check `tests/benchmarks/README.md` for detailed documentation.
**Want to run?** `uv run pytest -m benchmark tests/benchmarks/ -v`
