# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->
- market data repo unit test & IT
- Prompt: i want to create a load test for this service and measure execution time. lets say 100k randomly generated OHLCV data timeseries for the base timeseries timeframe, no yet cached higher timeframes data, freshly resampled to all timeframes up to the daily. measure the time and rate, how good or bad the current algo performs; identify flaws
- recreate preprocess service and integrate both resampling service and computing service
- circuit breakers

- archunit with hexagonal tests

## Future Epics (Unrefined)

- Performance optimization epic
- FTMO compliance validation
- Monitoring and observability
- Auto-scaling microservices

## Technical Debt

- cleanup common package
  - move strong business related to core
  - move adapter related to adapter

## Random Notes
<!-- Anything that doesn't fit elsewhere -->
