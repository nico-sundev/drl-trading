"""Dependency injection module for feature configuration writer in training service."""

import logging
from injector import Module, provider, singleton, Binder

from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_training.adapter.feature_config.feature_config_writer import FeatureConfigWriter

logger = logging.getLogger(__name__)


class TrainingFeatureConfigModule(Module):
    """Module providing feature configuration writer for training service."""

    def configure(self, binder: Binder) -> None:  # type: ignore[misc]
        """Configure interface bindings for feature configuration writer."""
        # Bind the write adapter specifically for training service
        binder.bind(FeatureConfigWriter, to=FeatureConfigWriter, scope=singleton)

    @provider
    @singleton
    def provide_feature_config_writer(
        self, session_factory: SQLAlchemySessionFactory
    ) -> FeatureConfigWriter:
        """Provide feature configuration writer adapter for training service."""
        return FeatureConfigWriter(session_factory)
