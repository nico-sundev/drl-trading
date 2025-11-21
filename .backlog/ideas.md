# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->

- resolve design issue with technical indicators and dask
- refactoring kafka_consumer_topic_adapter.py: move to adapter package and use DI to instantiate, remove from service bootstrap
- remove service_name and service_version from feature_store_config and use search for references
- test cov in adapter -> 40%
- remove in memory states from preprocessing and integrate redis as adapter
- apply repository docs writing best practises and align with my project goals
- featureversioninfoconfig use list[featuredefinition] instead of list[dict]
- refactor drl-trading-strategy-example in terms of hexarch design and fix violations (like the TypeMapper dependency of core -> adapter [feature -> indicator])
- prepare drl-trading-strategy-example for seamless replacement of proprietary package
- rename hexarch base module infrastructure ->  application
- archunit with hexagonal tests
- circuit breakers

Dask's processes scheduler uses multiprocessing to create separate Python processes
To send objects to worker processes, Python uses pickle to serialize them
threading.RLock cannot be pickled because locks are process-specific
Your TalippIndicatorService has self._lock = threading.RLock()
This service is injected into feature objects, making them unpicklable

## Future Epics (Unrefined)

- Performance optimization epic
- FTMO compliance validation
- Monitoring and observability
- Auto-scaling microservices

## Technical Debt

- rename infrastructure to application
- cleanup common package
  - move strong business related to core
  - move adapter related to adapter

## Random Notes
<!-- Anything that doesn't fit elsewhere -->
