from importlib.metadata import PackageNotFoundError, version

from drl_trading_common.config.logging_config import (
    configure_logging,  # pragma: no cover
)

from drl_trading_core.core_engine import CoreEngine

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Initialize logging
configure_logging()

__all__ = [
    "CoreEngine"
]
