"""Modern dependency injection container using injector library."""

import logging

from injector import Module

logger = logging.getLogger(__name__)


class CoreModule(Module):
    """Main application module for dependency injection.

    This module provides configuration values, complex factory logic, and interface bindings.
    Services with @inject decorators are auto-wired through interface bindings.
    """

    def __init__(self) -> None:
        ...
