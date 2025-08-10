from importlib.metadata import PackageNotFoundError, version

from drl_trading_common.logging.service_logger import ServiceLogger  # pragma: no cover

from drl_trading_core.core_engine import CoreEngine

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Initialize logging using default stage ('local') for library import context
ServiceLogger(service_name="drl-trading-core", stage="local").configure()

__all__ = [
    "CoreEngine"
]
