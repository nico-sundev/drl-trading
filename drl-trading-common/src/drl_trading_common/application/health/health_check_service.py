"""
Health check service for managing and executing health checks.

Coordinates multiple health checks and provides aggregated health status
for service monitoring and readiness probes.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .health_check import HealthCheck, HealthStatus

logger = logging.getLogger(__name__)


class HealthCheckService:
    """Service for managing and executing health checks."""

    def __init__(self):
        """Initialize health check service."""
        self.checks: List[HealthCheck] = []

    def register_check(self, check: HealthCheck) -> None:
        """
        Register a single health check.

        Args:
            check: HealthCheck instance to register
        """
        self.checks.append(check)
        logger.info(f"Registered health check: {check.get_name()}")

    def register_checks(self, checks: List[HealthCheck]) -> None:
        """
        Register multiple health checks.

        Args:
            checks: List of HealthCheck instances to register
        """
        for check in checks:
            self.register_check(check)

    def check_health(self) -> Dict[str, Any]:
        """
        Execute all health checks and return aggregated status.

        Returns:
            Dictionary containing overall status and individual check results
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        failed_checks = []
        degraded_checks = []

        for check in self.checks:
            try:
                result = check.check()
                results[check.get_name()] = result

                status = result.get("status", HealthStatus.UNHEALTHY.value)

                if status == HealthStatus.UNHEALTHY.value:
                    overall_status = HealthStatus.UNHEALTHY
                    failed_checks.append(check.get_name())
                elif (
                    status == HealthStatus.DEGRADED.value
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED
                    degraded_checks.append(check.get_name())

            except Exception as e:
                logger.error(
                    f"Health check {check.get_name()} failed with exception: {e}"
                )
                results[check.get_name()] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Health check failed with exception: {str(e)}",
                }
                overall_status = HealthStatus.UNHEALTHY
                failed_checks.append(check.get_name())

        # Build summary message
        summary_message = self._build_summary_message(
            overall_status, failed_checks, degraded_checks
        )

        return {
            "status": overall_status.value,
            "message": summary_message,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.checks),
            "failed_checks": failed_checks,
            "degraded_checks": degraded_checks,
        }

    def check_readiness(self) -> Dict[str, Any]:
        """
        Check service readiness (subset of health checks for readiness probes).

        For now, this is the same as health check, but can be customized
        to only include critical checks for service readiness.

        Returns:
            Dictionary containing readiness status
        """
        health_result = self.check_health()

        # For readiness, we might be more strict
        # Only consider service ready if ALL checks are healthy
        is_ready = health_result["status"] == HealthStatus.HEALTHY.value

        return {
            "ready": is_ready,
            "status": health_result["status"],
            "message": health_result["message"],
            "timestamp": health_result["timestamp"],
        }

    def get_registered_checks(self) -> List[str]:
        """
        Get list of registered health check names.

        Returns:
            List of health check names
        """
        return [check.get_name() for check in self.checks]

    def _build_summary_message(
        self,
        overall_status: HealthStatus,
        failed_checks: List[str],
        degraded_checks: List[str],
    ) -> str:
        """Build a summary message based on check results."""
        if overall_status == HealthStatus.HEALTHY:
            return "All health checks passing"
        elif overall_status == HealthStatus.DEGRADED:
            return f"Service degraded. Degraded checks: {', '.join(degraded_checks)}"
        else:
            message = f"Service unhealthy. Failed checks: {', '.join(failed_checks)}"
            if degraded_checks:
                message += f". Degraded checks: {', '.join(degraded_checks)}"
            return message
