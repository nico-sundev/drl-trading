
"""Deprecated in core: concrete FeatureStoreWrapper removed.

The concrete Feast-dependent wrapper now lives in adapter layer. This file is kept
as a minimal shim to avoid import breakage during transition. Remove once all
imports updated to adapter implementation or granular ports.
"""
from __future__ import annotations

class FeatureStoreWrapper:  # pragma: no cover - transitional shim
    def __init__(self, *args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError(
            "FeatureStoreWrapper has moved to adapter layer. Update DI bindings to use adapter implementation."
        )
