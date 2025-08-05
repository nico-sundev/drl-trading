"""
Health check framework for DRL Trading services.

Provides standardized health check patterns for monitoring service health
and readiness across all deployable services.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from enum import Enum
from datetime import datetime


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout_seconds: int = 5):
        """
        Initialize health check.

        Args:
            name: Name of the health check
            timeout_seconds: Maximum time to wait for check completion
        """
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dict with 'status', 'message', and optional details

        Example:
            {
                "status": "healthy",
                "message": "Database connection successful",
                "response_time_ms": 45,
                "details": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        """
        pass

    def get_name(self) -> str:
        """Get the name of this health check."""
        return self.name


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""

    def __init__(self, db_service, name: str = "database"):
        """
        Initialize database health check.

        Args:
            db_service: Database service with a ping() method
            name: Name for this health check
        """
        super().__init__(name)
        self.db_service = db_service

    def check(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            start_time = datetime.utcnow()
            self.db_service.ping()
            end_time = datetime.utcnow()

            response_time = int((end_time - start_time).total_seconds() * 1000)

            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Database connection successful",
                "response_time_ms": response_time
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Database connection failed: {str(e)}"
            }


class MessageBusHealthCheck(HealthCheck):
    """Health check for message bus connectivity."""

    def __init__(self, message_bus, name: str = "message_bus"):
        """
        Initialize message bus health check.

        Args:
            message_bus: Message bus service with health check capability
            name: Name for this health check
        """
        super().__init__(name)
        self.message_bus = message_bus

    def check(self) -> Dict[str, Any]:
        """Check message bus connectivity."""
        try:
            # Assuming message bus has some kind of health check method
            if hasattr(self.message_bus, 'is_connected'):
                if self.message_bus.is_connected():
                    return {
                        "status": HealthStatus.HEALTHY.value,
                        "message": "Message bus connection healthy"
                    }
                else:
                    return {
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": "Message bus not connected"
                    }
            else:
                # If no specific health check, assume healthy if object exists
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "Message bus service available"
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Message bus health check failed: {str(e)}"
            }


class ModelHealthCheck(HealthCheck):
    """Health check for ML model availability."""

    def __init__(self, model_service, name: str = "model"):
        """
        Initialize model health check.

        Args:
            model_service: Model service with health check capability
            name: Name for this health check
        """
        super().__init__(name)
        self.model_service = model_service

    def check(self) -> Dict[str, Any]:
        """Check model availability and readiness."""
        try:
            if hasattr(self.model_service, 'is_ready'):
                if self.model_service.is_ready():
                    return {
                        "status": HealthStatus.HEALTHY.value,
                        "message": "Model loaded and ready"
                    }
                else:
                    return {
                        "status": HealthStatus.DEGRADED.value,
                        "message": "Model loading in progress"
                    }
            else:
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "Model service available"
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Model health check failed: {str(e)}"
            }


class FeatureStoreHealthCheck(HealthCheck):
    """Health check for feature store connectivity."""

    def __init__(self, feature_store, name: str = "feature_store"):
        """
        Initialize feature store health check.

        Args:
            feature_store: Feature store service
            name: Name for this health check
        """
        super().__init__(name)
        self.feature_store = feature_store

    def check(self) -> Dict[str, Any]:
        """Check feature store connectivity."""
        try:
            # Try to access feature store
            if hasattr(self.feature_store, 'health_check'):
                result = self.feature_store.health_check()
                return {
                    "status": HealthStatus.HEALTHY.value if result else HealthStatus.UNHEALTHY.value,
                    "message": "Feature store available" if result else "Feature store unavailable"
                }
            else:
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "Feature store service available"
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Feature store health check failed: {str(e)}"
            }
