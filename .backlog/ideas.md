# Ideas & Brain Dump

## Quick Thoughts

- proceed with the plan, but explain the constraints of coupled features
- resolve design issue with technical indicators and dask
Dask's processes scheduler uses multiprocessing to create separate Python processes
To send objects to worker processes, Python uses pickle to serialize them
threading.RLock cannot be pickled because locks are process-specific
Your TalippIndicatorService has self._lock = threading.RLock()
This service is injected into feature objects, making them unpicklable
- prepare pyproject.toml of preprocessing and training project to integrate drl-trading-strategy seamlessly
  - priority: choose dependency from personal gitlab artifactory over workspace artifact
  - goal:
    - by default, after cloning the whole repository, the workspace's "drl-trading-strategy-example" should be used for compilation
    - for myself, i have the need for an easy switch as i am deploying my system with the "proprietary" version of the "drl-trading-strategy"
    - maybe solution (challenge this): use same dependency (name of the package), just distinct by version: the example project is being installed as -SNAPSHOT whereas my real one uses reasonable semvers
    - this should be somehow controllable over environment variable or some compareable config file, so i can populate that config switch for my personal stages and other people dont have it after freshly cloning the repo -> use snapshot artifact
- refactor drl-trading-strategy-example in terms of hexarch design and fix violations (like the TypeMapper dependency of core -> adapter [feature -> indicator])
- replace all primitive dataframes which are passed around from core to preprocessing by objects, holding these with built in validation logic e.g. for datetimeindex (Note: feast requires the event_timestamp column, whereas the rest of the application should be decoupled of this mandate - so a design for mapping column names from core to feast is also necessary)
- remove in memory states from preprocessing and integrate redis as adapter
- refactoring kafka_consumer_topic_adapter.py: move to adapter package and use DI to instantiate, remove from service bootstrap
- test cov in adapter too low -> 40%
- integrate archunit tests to test hexagonal achitecture
- design abstract resiliency patters: combine tenacity retry logic with python circuit breakers
- cleanup common package
  - leave common config stuff in there
- remove service_name and service_version from feature_store_config and use search for references
- reconsider naming standard for each services module: "infrastructure" -> "application"
- apply repository docs writing best practises and align with my project goals
  - goal: every documentation in this repository aligns with an individually forged doc-writing-guide; every doc has its purpose and dedicated audience -> some are high level, some low-level, some design, some about coding practises applied
  - context: every time, an LLM is helping with documentation, it becomes blown up, exaggerated and hard to read, also because of length
- refactor the whole system so not only OHLCV data is possible source for features, but also other kind of structured data (Context: in future we may want more than market data for features, for instance news data as timeseries or earning report data)
