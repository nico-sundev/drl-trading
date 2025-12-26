# Ideas & Brain Dump

## Quick Thoughts

- **Catchup Flow - Config-Driven Topic Routing**: Configure service config to map ProcessingContext.CATCHUP to completion topic "completed.preprocess-data.catchup" (avoid hardcoded logic in publishers)
- **Catchup Flow - State Persistence**: Implement catchup state tracking for crash recovery (Redis/DB: dataset_id, status, last_processed_timestamp, warmup_satisfied, started_at, completed_at)
- **Catchup Flow - Memory Optimization**: Implement chunked processing for large catchups (>1M records, multiple symbols) to prevent OOM
- **Catchup Flow - Idempotency**: Add idempotency handling with Redis to allow safe retry of catchup requests without duplicate work
- go on with e2e training test
- refactoring the interface between drl-trading-core and drl-trading-strategy
  - currently: 100% decoupling, both depend on interfaces in drl-trading-common, coupling happens in service where both are needed (preprocessing), directly in injector setup
  - future expectation: treat drl-trading-strategy like an adapter, following existing hexarch design
    - move common interfaces away from drl-trading-common into drl-trading-core
    - create dependency of strategy->core
    - move implementations of strategy package, like featurefactory or featureregistry/featureconfigregistry into core
    - evaluate best way to tie together both packages practically using DI injector (use drl-trading-preprocess to examine)
- refactor drl-trading-strategy-example in terms of hexarch design and fix violations (like the TypeMapper dependency of core -> adapter [feature -> indicator])
- refactoring kafka_consumer_topic_adapter.py: move to adapter package and use DI to instantiate, remove from service bootstrap
- design abstract resiliency patters: combine tenacity retry logic with python circuit breakers
- cleanup common package
  - leave common config stuff in there
- remove hardcoded offline/online config setting for featureview creation in feast_provider.py and replace by dynamic configuration
- remove hardcoded "symbol" column in feast_provider.py and replace by a config array
  - also consider this in JOIN KEYS Field creation in feast_provider.py
- remove service_name and service_version from feature_store_config and use search for references
- implement Redis caching for OHLCV availability checks in FeatureCoverageAnalyzer to eliminate redundant database calls when processing multiple symbols/timeframes (e.g., training request with 10 symbols × 3 timeframes = 30 redundant base timeframe checks → cache with TTL)
- refactor the whole system so not only OHLCV data is possible source for features, but also other kind of structured data (Context: in future we may want more than market data for features, for instance news data as timeseries or earning report data)
