"""Resilience configuration for retry policies and error handling patterns.

This module provides generic, reusable configuration schemas for retry logic,
circuit breakers, and other resilience patterns. Services define their own
use case-specific constants and mappings to these configurations.
"""

from typing import Dict

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Configuration for exponential backoff retry logic.

    This configuration is used by resilience decorators (e.g., tenacity)
    to implement retry behavior with exponential backoff and jitter.

    Attributes:
        max_attempts: Maximum number of retry attempts before giving up.
        wait_exponential_multiplier: Multiplier for exponential backoff (seconds).
            Formula: wait = multiplier * (2 ^ attempt_number)
        wait_exponential_max: Maximum wait time between retries (seconds).
        wait_jitter_max: Maximum random jitter to add to wait time (seconds).
            Helps prevent thundering herd when many clients retry simultaneously.
        stop_after_delay: Optional total time limit for all retries (seconds).
            If set, stops retrying after this cumulative time, regardless of max_attempts.
    """

    max_attempts: int = Field(
        ge=1,
        description="Maximum number of retry attempts (including initial attempt)",
    )
    wait_exponential_multiplier: float = Field(
        ge=0.0,
        description="Multiplier for exponential backoff calculation (seconds)",
    )
    wait_exponential_max: float = Field(
        ge=0.0,
        description="Maximum wait time between retries (seconds)",
    )
    wait_jitter_max: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum random jitter to add to wait time (seconds)",
    )
    stop_after_delay: float | None = Field(
        default=None,
        ge=0.0,
        description="Total time limit for all retries (seconds, optional)",
    )


class ResilienceConfig(BaseModel):
    """Container for all resilience configurations in a service.

    Services define individual retry policies with explicit keys and reference
    them by constant. This allows per-use-case configuration flexibility.

    Example:
        ```yaml
        infrastructure:
          resilience:
            retry_configs:
              kafka_resampled_data_retry:
                max_attempts: 5
                wait_exponential_multiplier: 1.0
                wait_exponential_max: 60.0
              kafka_dlq_retry:
                max_attempts: 1
                wait_exponential_multiplier: 1.0
                wait_exponential_max: 1.0
        ```

    Attributes:
        retry_configs: Flat registry of retry configurations keyed by use case.
            Services define the keys via constants (e.g., RETRY_CONFIG_KAFKA_RESAMPLED_DATA).
    """

    retry_configs: Dict[str, RetryConfig] = Field(
        default_factory=dict,
        description="Registry of retry configurations by use case key",
    )

    def get_retry_config(self, key: str) -> RetryConfig:
        """Retrieve a retry configuration by key.

        Args:
            key: The use case-specific key for the retry config.
                Should be defined as a constant in the service's config module.

        Returns:
            The retry configuration for the specified key.

        Raises:
            KeyError: If the key is not found in the registry.
                This typically indicates a misconfiguration or missing YAML entry.
        """
        if key not in self.retry_configs:
            raise KeyError(
                f"Retry config key '{key}' not found in resilience configuration. "
                f"Available keys: {list(self.retry_configs.keys())}"
            )
        return self.retry_configs[key]
