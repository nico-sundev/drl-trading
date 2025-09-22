# Load Tests

Simple performance tests for MarketDataResamplingService.

## Running Tests

```bash
# Run all load tests
pytest tests/simple_load_test.py -m load -v

# Run comprehensive scaling analysis
pytest tests/comprehensive_load_test.py -m load -v

# Run specific test
pytest tests/simple_load_test.py::TestLoadPerformance::test_100k_records_performance -m load -v
```

## Test Results

See `LOAD_TEST_RESULTS.md` for detailed performance analysis.

**Summary**:
- **Processing Rate**: 7k-17k records/second
- **Memory Usage**: ~1.4GB per 1M records
- **Scaling**: Excellent O(n) complexity
- **Limit**: ~8M records before memory pressure
