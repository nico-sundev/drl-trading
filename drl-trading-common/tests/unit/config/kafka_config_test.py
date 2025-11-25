"""Unit tests for Kafka configuration classes."""

import pytest

from drl_trading_common.config.kafka_config import (
    ConsumerFailurePolicy,
    TopicSubscription,
)


class TestConsumerFailurePolicy:
    """Unit tests for ConsumerFailurePolicy schema validation."""

    def test_default_values(self) -> None:
        """Test that default values are applied correctly."""
        # Given / When
        policy = ConsumerFailurePolicy()

        # Then
        assert policy.max_retries == 3  # Default retries
        assert policy.dlq_topic is None
        assert policy.track_retry_in_headers is True

    def test_custom_values(self) -> None:
        """Test creating policy with custom values."""
        # Given / When
        policy = ConsumerFailurePolicy(
            max_retries=5,
            dlq_topic="dlq.my-topic",
            track_retry_in_headers=False,
        )

        # Then
        assert policy.max_retries == 5
        assert policy.dlq_topic == "dlq.my-topic"
        assert policy.track_retry_in_headers is False

    def test_max_retries_cannot_be_negative(self) -> None:
        """Test that max_retries validation rejects negative values."""
        # Given / When / Then
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            ConsumerFailurePolicy(max_retries=-1)

    def test_zero_max_retries_is_valid(self) -> None:
        """Test that zero max_retries (infinite retries) is valid."""
        # Given / When
        policy = ConsumerFailurePolicy(max_retries=0)

        # Then
        assert policy.max_retries == 0

    def test_policy_with_dlq_topic(self) -> None:
        """Test policy with DLQ topic configured."""
        # Given / When
        policy = ConsumerFailurePolicy(
            max_retries=3,
            dlq_topic="dlq.critical-data",
        )

        # Then
        assert policy.max_retries == 3
        assert policy.dlq_topic == "dlq.critical-data"
        assert policy.track_retry_in_headers is True  # Default

    def test_policy_without_dlq_topic(self) -> None:
        """Test policy without DLQ topic (messages lost after max retries)."""
        # Given / When
        policy = ConsumerFailurePolicy(
            max_retries=1,
            dlq_topic=None,
        )

        # Then
        assert policy.max_retries == 1
        assert policy.dlq_topic is None


class TestTopicSubscription:
    """Unit tests for TopicSubscription schema."""

    def test_required_fields(self) -> None:
        """Test that topic and handler_id are required."""
        # Given / When
        subscription = TopicSubscription(
            topic="my-topic",
            handler_id="my_handler",
        )

        # Then
        assert subscription.topic == "my-topic"
        assert subscription.handler_id == "my_handler"
        assert subscription.failure_policy_key is None  # Optional

    def test_with_failure_policy_key(self) -> None:
        """Test subscription with failure policy reference."""
        # Given / When
        subscription = TopicSubscription(
            topic="critical.data",
            handler_id="critical_handler",
            failure_policy_key="critical_data_policy",
        )

        # Then
        assert subscription.topic == "critical.data"
        assert subscription.handler_id == "critical_handler"
        assert subscription.failure_policy_key == "critical_data_policy"

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation error."""
        # Given / When / Then
        with pytest.raises(ValueError):
            TopicSubscription()  # type: ignore[call-arg]
