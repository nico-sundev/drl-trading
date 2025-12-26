"""
Basic health checks for DRL Trading services.

Provides common health check implementations that can be used
across all microservices.
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional

from drl_trading_common.application.health.health_check import (
    HealthCheck,
    HealthStatus,
)

logger = logging.getLogger(__name__)


class SystemResourcesHealthCheck(HealthCheck):
    """
    Health check for system resources (CPU, Memory).

    Monitors basic system health indicators.
    """

    def __init__(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        name: str = "system_resources",
    ):
        """
        Initialize system resources health check.

        Args:
            cpu_threshold: CPU usage percentage threshold for degraded status
            memory_threshold: Memory usage percentage threshold for degraded status
            name: Name of this health check
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.name = name

    def get_name(self) -> str:
        """Get the name of this health check."""
        return self.name

    def check(self) -> Dict[str, Any]:
        """
        Check system resource health.

        Returns:
            Dictionary with health status and resource metrics
        """
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Determine overall status
            status = HealthStatus.HEALTHY
            messages = []

            if cpu_percent > self.cpu_threshold:
                status = (
                    HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status
                )
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory_percent > self.memory_threshold:
                status = (
                    HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status
                )
                messages.append(f"High memory usage: {memory_percent:.1f}%")

            message = "; ".join(messages) if messages else "System resources healthy"

            return {
                "status": status.value,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "thresholds": {
                        "cpu_threshold": self.cpu_threshold,
                        "memory_threshold": self.memory_threshold,
                    },
                },
            }

        except Exception as e:
            logger.error(f"System resources health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Failed to check system resources: {str(e)}",
            }


class ServiceStartupHealthCheck(HealthCheck):
    """
    Health check that tracks service startup completion.

    Useful for readiness probes to ensure service is fully initialized.
    """

    def __init__(self, name: str = "service_startup"):
        """
        Initialize service startup health check.

        Args:
            name: Name of this health check
        """
        self.name = name
        self.startup_completed = False
        self.startup_time: Optional[float] = None
        self.startup_errors: List[str] = []

    def get_name(self) -> str:
        """Get the name of this health check."""
        return self.name

    def mark_startup_completed(
        self, success: bool = True, error_message: Optional[str] = None
    ) -> None:
        """
        Mark service startup as completed.

        Args:
            success: Whether startup was successful
            error_message: Optional error message if startup failed
        """
        self.startup_completed = True
        self.startup_time = time.time()

        if not success and error_message:
            self.startup_errors.append(error_message)

    def check(self) -> Dict[str, Any]:
        """
        Check service startup status.

        Returns:
            Dictionary with startup health status
        """
        if not self.startup_completed:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": "Service startup not completed",
            }

        if self.startup_errors:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Service startup failed: {'; '.join(self.startup_errors)}",
            }

        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Service startup completed successfully",
            "details": {
                "startup_time": self.startup_time,
                "uptime_seconds": (
                    time.time() - self.startup_time if self.startup_time else 0
                ),
            },
        }


class ConfigurationHealthCheck(HealthCheck):
    """
    Health check that validates service configuration.

    Ensures critical configuration values are present and valid.
    """

    def __init__(self, config, name: str = "configuration"):
        """
        Initialize configuration health check.

        Args:
            config: Configuration object to validate
            name: Name of this health check
        """
        self.config = config
        self.name = name

    def get_name(self) -> str:
        """Get the name of this health check."""
        return self.name

    def check(self) -> Dict[str, Any]:
        """
        Check configuration health.

        Returns:
            Dictionary with configuration health status
        """
        try:
            if not self.config:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Configuration not loaded",
                }

            # Basic validation - ensure required attributes exist
            required_attrs = ["app_name", "stage"]
            missing_attrs = []

            for attr in required_attrs:
                if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                    missing_attrs.append(attr)

            if missing_attrs:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Missing required configuration attributes: {', '.join(missing_attrs)}",
                }

            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Configuration is valid",
                "details": {
                    "app_name": getattr(self.config, "app_name", "unknown"),
                    "stage": getattr(self.config, "stage", "unknown"),
                    "version": getattr(self.config, "version", "unknown"),
                },
            }

        except Exception as e:
            logger.error(f"Configuration health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Configuration validation failed: {str(e)}",
            }
