"""Retry metadata constants and utilities for Kafka message retry mechanism."""

from datetime import datetime, timezone

# Retry metadata header keys
HEADER_RETRY_ATTEMPT = "x-retry-attempt"
HEADER_ORIGINAL_TOPIC = "x-original-topic"
HEADER_FIRST_FAILURE_TIMESTAMP = "x-first-failure-timestamp"
HEADER_LAST_FAILURE_TIMESTAMP = "x-last-failure-timestamp"
HEADER_ERROR_TYPE = "x-error-type"
HEADER_ERROR_MESSAGE = "x-error-message"
HEADER_NEXT_RETRY_AFTER = "x-next-retry-after"


def extract_retry_attempt(headers: list[tuple[str, bytes]] | None) -> int:
    """
    Extract retry attempt count from message headers.

    Args:
        headers: Kafka message headers as list of (key, value) tuples.

    Returns:
        Retry attempt number (0 if header not found).
    """
    if not headers:
        return 0

    for key, value in headers:
        if key == HEADER_RETRY_ATTEMPT:
            try:
                return int(value.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return 0

    return 0


def extract_header_value(headers: list[tuple[str, bytes]] | None, header_key: str) -> str | None:
    """
    Extract string value from message headers.

    Args:
        headers: Kafka message headers as list of (key, value) tuples.
        header_key: Header key to look for.

    Returns:
        Header value as string, or None if not found.
    """
    if not headers:
        return None

    for key, value in headers:
        if key == header_key:
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return None

    return None


def build_retry_headers(
    retry_attempt: int,
    original_topic: str,
    error_type: str,
    error_message: str,
    first_failure_timestamp: str | None = None,
    next_retry_after_seconds: float | None = None,
) -> list[tuple[str, bytes]]:
    """
    Build retry metadata headers for publishing to retry topic.

    Args:
        retry_attempt: Current retry attempt number (1-indexed).
        original_topic: Topic where message originally came from.
        error_type: Exception class name.
        error_message: Error description.
        first_failure_timestamp: ISO timestamp of first failure (optional).
        next_retry_after_seconds: Seconds to wait before retry (optional).

    Returns:
        List of (header_key, header_value) tuples ready for Kafka publish.
    """
    now = datetime.now(timezone.utc).isoformat()

    headers: list[tuple[str, bytes]] = [
        (HEADER_RETRY_ATTEMPT, str(retry_attempt).encode("utf-8")),
        (HEADER_ORIGINAL_TOPIC, original_topic.encode("utf-8")),
        (HEADER_LAST_FAILURE_TIMESTAMP, now.encode("utf-8")),
        (HEADER_ERROR_TYPE, error_type.encode("utf-8")),
        (HEADER_ERROR_MESSAGE, error_message[:500].encode("utf-8")),  # Truncate long messages
    ]

    if first_failure_timestamp:
        headers.append((HEADER_FIRST_FAILURE_TIMESTAMP, first_failure_timestamp.encode("utf-8")))
    else:
        headers.append((HEADER_FIRST_FAILURE_TIMESTAMP, now.encode("utf-8")))

    if next_retry_after_seconds is not None:
        retry_after_timestamp = datetime.now(timezone.utc).timestamp() + next_retry_after_seconds
        headers.append((HEADER_NEXT_RETRY_AFTER, str(retry_after_timestamp).encode("utf-8")))

    return headers


def calculate_backoff_seconds(
    retry_attempt: int,
    base_seconds: float,
    multiplier: float,
    max_seconds: float = 300.0,
) -> float:
    """
    Calculate exponential backoff delay for retry attempt.

    Formula: min(base_seconds * (multiplier ^ (retry_attempt - 1)), max_seconds)

    Args:
        retry_attempt: Current retry attempt (1-indexed).
        base_seconds: Base delay for first retry.
        multiplier: Exponential multiplier (e.g., 2.0 = double each time).
        max_seconds: Maximum backoff cap (default 5 minutes).

    Returns:
        Backoff delay in seconds.

    Examples:
        >>> calculate_backoff_seconds(1, 1.0, 2.0)  # 1st retry
        1.0
        >>> calculate_backoff_seconds(2, 1.0, 2.0)  # 2nd retry
        2.0
        >>> calculate_backoff_seconds(3, 1.0, 2.0)  # 3rd retry
        4.0
    """
    if retry_attempt <= 0:
        return 0.0

    delay = base_seconds * (multiplier ** (retry_attempt - 1))
    return min(delay, max_seconds)
