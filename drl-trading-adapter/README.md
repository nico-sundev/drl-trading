## DRL Trading Adapter Package

Purpose: Houses external adapter implementations decoupled from core domain logic.

Design goals:
- Core owns ports (interfaces) and business logic.
- Adapter implements those ports using external frameworks (Feast, S3/local storage, etc.).
- Strategy package remains a pluggable external dependency providing proprietary feature definitions and gym environment.

Initial scope:
- Feast feature store provider (FeatureStoreWrapper, FeastProvider)
- Offline feature repositories (local filesystem, S3) via strategy selection
- Feature store save/fetch repositories orchestrating Feast apply/materialize operations

Planned additions:
- Additional messaging/database adapters as they are extracted from services
- Health check/web adapters if they become shared

Feast integration notes:
- Feast is optional at system level; enable/disable via FeatureStoreConfig.
- Offline repo strategy (local vs S3) selected by configuration; adapter implements both paths.
- Core depends only on abstract ports (e.g., IOfflineFeatureRepository) defined in common/core; concrete classes reside here.

Proprietary strategy package:
- Lives outside this repository; acts as an external adapter that supplies concrete feature objects and environment implementation.
- The example strategy inside this repo is a stub for integration and demonstration only.

Next steps:
1. Migrate existing Feast-related implementations from core into this package (in progress).
2. Adjust DI wiring so services bind core ports to adapter implementations.
3. Add targeted tests (unit + integration) once code stabilized post-move.

Versioning:
- Follows semantic versioning; initial pre-1.0 phase while extraction stabilizes.
