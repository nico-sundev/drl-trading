"""Dependency injection module for feature configuration adapters."""

import logging
from injector import Module, provider, singleton, Binder

from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_adapter.adapter.feature_config.feature_config_repository import FeatureConfigRepository
from drl_trading_core.core.port.feature_config_reader_port import FeatureConfigReaderPort

logger = logging.getLogger(__name__)


class FeatureConfigModule(Module):
    """Module providing feature configuration read adapter for all services."""

    def configure(self, binder: Binder) -> None:  # type: ignore[misc]
        """Configure interface bindings for feature configuration components."""
        # Bind the read adapter for all services that need to read feature configs
        binder.bind(FeatureConfigReaderPort, to=FeatureConfigRepository, scope=singleton)  # type: ignore[misc]

    @provider
    @singleton
    def provide_feature_config_repository(
        self, session_factory: SQLAlchemySessionFactory
    ) -> FeatureConfigRepository:
        """Provide feature configuration repository adapter."""
        return FeatureConfigRepository(session_factory)
