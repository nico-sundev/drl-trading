# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->

- create e2e test for preprocess service
- remove service_name and service_version from feature_store_config and use search for references
- test cov in adapter -> 40%
- preprocessing IT is almost passing ... check if features really have to be passed to the adapter and
  - if so, then create a response object from featuremanager, adding the features
- apply repository docs writing best practises and align with my project goals
- create new config class for feature manager: batch size for concurrent feature computation
- drltradingadapter contains both test and tests directories
- featureversioninfoconfig use list[featuredefinition] instead of list[dict]
- prepare drl-trading-strategy-example for seamless replacement of proprietary package
- archunit with hexagonal tests
- circuit breakers

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
