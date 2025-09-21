# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->
- market data repo unit test & IT
- Prompt: i want to create a load test for this service and measure execution time. lets say 100k randomly generated OHLCV data timeseries for the base timeseries timeframe, resampled to all timeframes up to the daily. measure the time and rate, how good or bad the algo performs
- recreate preprocess service and integrate both resampling service and computing service
- circuit breakers
- cleanup common package
    - move strong business related to core
    - move adapter related to adapter
- archunit with hexagonal tests

## Future Epics (Unrefined)

- Performance optimization epic
- FTMO compliance validation
- Monitoring and observability
- Auto-scaling microservices

## Technical Debt

- Consolidate configuration patterns
- Improve error handling consistency
- Add more comprehensive logging

## Random Notes
<!-- Anything that doesn't fit elsewhere -->
