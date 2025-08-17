"""Bootstrap logging utilities.

Provides a lightweight, temporary logger for very early startup phases
before the full `ServiceLogger` configuration is applied. This allows
emitting diagnostic messages about configuration loading and dependency
wiring without duplicating the full logging setup logic or risking
`basicConfig` side effects.

Usage pattern:

    from drl_trading_common.logging.bootstrap_logging import (
        get_bootstrap_logger, retire_bootstrap_logger
    )

    bootstrap_logger = get_bootstrap_logger(service_name)
    bootstrap_logger.info("Loading configuration...")
    # load / validate config
    service_logger = ServiceLogger(service_name, stage, config)
    service_logger.configure()
    retire_bootstrap_logger(service_name)
    logger = service_logger.get_logger()
    logger.info("Startup continuing with full logging stack")

The helper is intentionally minimal: single StreamHandler, simple format,
no integration with short name abbreviation (that comes with full config).
"""
from __future__ import annotations

import logging

_BOOTSTRAP_HANDLER_ATTR = "_bootstrap_handler_installed"


def get_bootstrap_logger(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a temporary bootstrap logger for a service.

    Logger name pattern: ``{service_name}.bootstrap``. A handler is only
    attached once; subsequent calls return the same configured logger.
    """
    name = f"{service_name}.bootstrap"
    logger = logging.getLogger(name)
    if not getattr(logger, _BOOTSTRAP_HANDLER_ATTR, False):  # type: ignore[attr-defined]
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | BOOT | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        setattr(logger, _BOOTSTRAP_HANDLER_ATTR, True)
    return logger


def retire_bootstrap_logger(service_name: str) -> None:
    """Remove handlers installed by `get_bootstrap_logger`.

    Safe to call multiple times. After retirement, subsequent calls
    to `get_bootstrap_logger` will re-install a handler.
    """
    name = f"{service_name}.bootstrap"
    logger = logging.getLogger(name)
    if getattr(logger, _BOOTSTRAP_HANDLER_ATTR, False):  # type: ignore[attr-defined]
        for h in list(logger.handlers):
            logger.removeHandler(h)
        setattr(logger, _BOOTSTRAP_HANDLER_ATTR, False)


__all__ = ["get_bootstrap_logger", "retire_bootstrap_logger"]
