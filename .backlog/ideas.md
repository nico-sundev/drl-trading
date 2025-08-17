# Ideas & Brain Dump

## Quick Thoughts
<!-- Dump ideas here quickly, organize later -->
### Check logs
- starting service current works without errors, even if stage envvar not provided

### New Package
- drl-trading-adapter where adapter logic from current drl-trading-core resides

### Separate domain from adapter logic for feast
- Upgrade current integration of feast FeatureStore to be hexagonal architecture confirm
- Below services contain mostly domain logic, this should be abstracted from adapter logic
  - feast_provider.py
  - feature_store_fetch_repo.py
  - feature_store_store_repo.py
- Utilize existing interface abstractions, try to keep it and adapt on it

### Break up drl-trading-core`s services
-- Refactor core package, comply with hexagonal architecture
-- More specification is coming soon regards inter-service dependencies


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
