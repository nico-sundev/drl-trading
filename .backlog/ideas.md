# Ideas & Brain Dump

## Quick Thoughts

- implement Redis caching for OHLCV availability checks in FeatureCoverageAnalyzer to eliminate redundant database calls when processing multiple symbols/timeframes (e.g., training request with 10 symbols × 3 timeframes = 30 redundant base timeframe checks → cache with TTL)
- go on with e2e training test
- refactoring the interface between drl-trading-core and drl-trading-strategy
  - currently: 100% decoupling, both depend on interfaces in drl-trading-common, coupling happens in service where both are needed (preprocessing), directly in injector setup
  - future expectation: treat drl-trading-strategy like an adapter, following existing hexarch design
    - move common interfaces away from drl-trading-common into drl-trading-core
    - create dependency of strategy->core
    - move implementations of strategy package, like featurefactory or featureregistry/featureconfigregistry into core
    - evaluate best way to tie together both packages practically using DI injector (use drl-trading-preprocess to examine)
- refactor drl-trading-strategy-example in terms of hexarch design and fix violations (like the TypeMapper dependency of core -> adapter [feature -> indicator])
- replace all primitive dataframes which are passed around from core to preprocessing by objects, holding these with built in validation logic e.g. for datetimeindex (Note: feast requires the event_timestamp column, whereas the rest of the application should be decoupled of this mandate - so a design for mapping column names from core to feast is also necessary)
- remove in memory states from preprocessing and integrate redis as adapter
- refactoring kafka_consumer_topic_adapter.py: move to adapter package and use DI to instantiate, remove from service bootstrap
- test cov in adapter too low -> 40%
- integrate archunit tests to test hexagonal achitecture
- design abstract resiliency patters: combine tenacity retry logic with python circuit breakers
- cleanup common package
  - leave common config stuff in there
- remove hardcoded offline/online config setting for featureview creation in feast_provider.py and replace by dynamic configuration
- remove hardcoded "symbol" column in feast_provider.py and replace by a config array
  - also consider this in JOIN KEYS Field creation in feast_provider.py
- remove service_name and service_version from feature_store_config and use search for references
- reconsider naming standard for each services module: "infrastructure" -> "application"
- apply repository docs writing best practises and align with my project goals
  - goal: every documentation in this repository aligns with an individually forged doc-writing-guide; every doc has its purpose and dedicated audience -> some are high level, some low-level, some design, some about coding practises applied
  - context: every time, an LLM is helping with documentation, it becomes blown up, exaggerated and hard to read, also because of length
- refactor the whole system so not only OHLCV data is possible source for features, but also other kind of structured data (Context: in future we may want more than market data for features, for instance news data as timeseries or earning report data)
