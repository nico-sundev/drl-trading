# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->

scenarios:
- go on with the feature existence check port (remove)
1. compute features for a timespan, which is covered by feast -> return features by feast [training_only]
2. compute features for a timespan, which not yet covered by feast -> make sure, the respective features (subclasses of basefeature) are being warmed up (initialized with lets say last 500 OHLCV data) up to the beginning of the desired timespan for feature computation -> returned features by feast, which are in desired timespan -> compute the rest of the missing time period via computing service [training or inference]
3. compute features for the latest data record, which can for obvious reasons not exist in feast yet and has to be computed right now, but also needs warm up (same like the prior step, but only for the initialization of the preprocessing service, necessary to compute features for latest data live) [inference_only]

For training, catching up on the latest data can happen via a feature flag in trainingconfig, on demand. For inference, catching up MUST happen at every service start (responsibility of ingest service) and there has to be a check in preprocessing service, if ALL registered features have caught up in real time. practically, this is a major constraint, before a consistent inference is possible. if there were gaps in between of the data stream, we would compute features for timestamps, that were in the past, but not NOW.
Technically, ingest service gets asyncronously called with new data and it MUST send it as featurecomputerequest to preprocessing service. but unless the state (warmup) is exactly where it needs to be (computed all prior data), the inference can not proceed and is blocked.

Technically speaking ...
- to catch up with recent data means to ask the data provider for ALL data up to the most recent available data records, beginning from the last records available in our timescale db -> then store it into timescale
- live ingest means we get new OHLCV records every couple of minutes (depends on our timeframe config) and sends it to preprocessing service

- circuit breakers
- archunit with hexagonal tests

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
