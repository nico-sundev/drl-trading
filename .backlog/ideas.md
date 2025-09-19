# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->
- create resampling service in drl-trading-preprocess
    - in: dict of list of marketdatamodel of all timeframes
    - out: dict of list of marketdatamodel of all timeframes
    - scenarios:
        - first time:
            - only lowest TF may exist, no higher ones
            - start resampling lowest TF to each TF individually, beginning from 0
        - every other time:
            - multiple TF exist
            - lowest TF contains the latest timestamps
            - try to resample for each TF individually (either the new timeseries records can produce a new higher TF record or not yet)
    - save all newly produced resamplings into database
    - process with feature computation of each TFs timeseries
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
