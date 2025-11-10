"""Unit tests for retry metadata utilities."""

import pytest

from drl_trading_common.adapter.messaging.retry_metadata import (
    HEADER_ERROR_MESSAGE,
    HEADER_ERROR_TYPE,
    HEADER_FIRST_FAILURE_TIMESTAMP,
    HEADER_LAST_FAILURE_TIMESTAMP,
    HEADER_NEXT_RETRY_AFTER,
    HEADER_ORIGINAL_TOPIC,
    HEADER_RETRY_ATTEMPT,
    build_retry_headers,
    calculate_backoff_seconds,
    extract_header_value,
    extract_retry_attempt,
)


class TestExtractRetryAttempt:
    """Tests for extract_retry_attempt function."""

    def test_extract_retry_attempt_from_valid_headers(self) -> None:
        """Test extracting retry attempt from valid headers."""
        # Given
        headers = [
            ("x-retry-attempt", b"3"),
            ("x-original-topic", b"test.topic"),
        ]

        # When
        result = extract_retry_attempt(headers)

        # Then
        assert result == 3

    def test_extract_retry_attempt_returns_zero_when_header_missing(self) -> None:
        """Test returns 0 when retry attempt header is missing."""
        # Given
        headers = [
            ("x-original-topic", b"test.topic"),
            ("other-header", b"value"),
        ]

        # When
        result = extract_retry_attempt(headers)

        # Then
        assert result == 0

    def test_extract_retry_attempt_returns_zero_when_headers_none(self) -> None:
        """Test returns 0 when headers parameter is None."""
        # Given
        headers = None

        # When
        result = extract_retry_attempt(headers)

        # Then
        assert result == 0

    def test_extract_retry_attempt_returns_zero_when_value_invalid(self) -> None:
        """Test returns 0 when header value cannot be parsed as integer."""
        # Given
        headers = [
            ("x-retry-attempt", b"not-a-number"),
        ]

        # When
        result = extract_retry_attempt(headers)

        # Then
        assert result == 0

    def test_extract_retry_attempt_handles_unicode_decode_error(self) -> None:
        """Test returns 0 when header value has invalid UTF-8 encoding."""
        # Given
        headers = [
            ("x-retry-attempt", b"\xff\xfe"),  # Invalid UTF-8
        ]

        # When
        result = extract_retry_attempt(headers)

        # Then
        assert result == 0


class TestExtractHeaderValue:
    """Tests for extract_header_value function."""

    def test_extract_header_value_returns_string_when_found(self) -> None:
        """Test extracting string value from headers."""
        # Given
        headers = [
            ("x-error-type", b"ValueError"),
            ("x-error-message", b"Something went wrong"),
        ]

        # When
        result = extract_header_value(headers, "x-error-type")

        # Then
        assert result == "ValueError"

    def test_extract_header_value_returns_none_when_not_found(self) -> None:
        """Test returns None when header key not found."""
        # Given
        headers = [
            ("x-error-type", b"ValueError"),
        ]

        # When
        result = extract_header_value(headers, "x-missing-header")

        # Then
        assert result is None

    def test_extract_header_value_returns_none_when_headers_none(self) -> None:
        """Test returns None when headers parameter is None."""
        # Given
        headers = None

        # When
        result = extract_header_value(headers, "x-any-header")

        # Then
        assert result is None

    def test_extract_header_value_handles_unicode_decode_error(self) -> None:
        """Test returns None when header value has invalid UTF-8."""
        # Given
        headers = [
            ("x-error-type", b"\xff\xfe"),  # Invalid UTF-8
        ]

        # When
        result = extract_header_value(headers, "x-error-type")

        # Then
        assert result is None


class TestBuildRetryHeaders:
    """Tests for build_retry_headers function."""

    def test_build_retry_headers_with_minimal_params(self) -> None:
        """Test building retry headers with only required parameters."""
        # Given
        retry_attempt = 2
        original_topic = "requested.preprocess-data"
        error_type = "ValidationError"
        error_message = "Missing required field"

        # When
        headers = build_retry_headers(
            retry_attempt=retry_attempt,
            original_topic=original_topic,
            error_type=error_type,
            error_message=error_message,
        )

        # Then
        assert len(headers) == 6  # 6 headers total (including first_failure auto-generated)

        header_dict = {key: value.decode("utf-8") for key, value in headers}
        assert header_dict[HEADER_RETRY_ATTEMPT] == "2"
        assert header_dict[HEADER_ORIGINAL_TOPIC] == "requested.preprocess-data"
        assert header_dict[HEADER_ERROR_TYPE] == "ValidationError"
        assert header_dict[HEADER_ERROR_MESSAGE] == "Missing required field"
        assert HEADER_LAST_FAILURE_TIMESTAMP in header_dict
        assert HEADER_FIRST_FAILURE_TIMESTAMP in header_dict

    def test_build_retry_headers_with_all_params(self) -> None:
        """Test building retry headers with all optional parameters."""
        # Given
        retry_attempt = 1
        original_topic = "test.topic"
        error_type = "RuntimeError"
        error_message = "Test error"
        first_failure_timestamp = "2024-01-01T12:00:00"
        next_retry_after_seconds = 5.0

        # When
        headers = build_retry_headers(
            retry_attempt=retry_attempt,
            original_topic=original_topic,
            error_type=error_type,
            error_message=error_message,
            first_failure_timestamp=first_failure_timestamp,
            next_retry_after_seconds=next_retry_after_seconds,
        )

        # Then
        assert len(headers) == 7  # All headers including next_retry_after

        header_dict = {key: value.decode("utf-8") for key, value in headers}
        assert header_dict[HEADER_FIRST_FAILURE_TIMESTAMP] == first_failure_timestamp
        assert HEADER_NEXT_RETRY_AFTER in header_dict
        # Verify next_retry_after is a parseable timestamp
        float(header_dict[HEADER_NEXT_RETRY_AFTER])

    def test_build_retry_headers_truncates_long_error_message(self) -> None:
        """Test error message is truncated to 500 characters."""
        # Given
        long_message = "x" * 1000  # 1000 characters

        # When
        headers = build_retry_headers(
            retry_attempt=1,
            original_topic="test.topic",
            error_type="Error",
            error_message=long_message,
        )

        # Then
        header_dict = {key: value.decode("utf-8") for key, value in headers}
        assert len(header_dict[HEADER_ERROR_MESSAGE]) == 500


class TestCalculateBackoffSeconds:
    """Tests for calculate_backoff_seconds function."""

    def test_calculate_backoff_first_retry(self) -> None:
        """Test backoff calculation for first retry attempt."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=1,
            base_seconds=1.0,
            multiplier=2.0,
        )

        # Then
        assert result == 1.0  # base_seconds * (2.0 ^ 0)

    def test_calculate_backoff_second_retry(self) -> None:
        """Test backoff calculation for second retry attempt."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=2,
            base_seconds=1.0,
            multiplier=2.0,
        )

        # Then
        assert result == 2.0  # base_seconds * (2.0 ^ 1)

    def test_calculate_backoff_third_retry(self) -> None:
        """Test backoff calculation for third retry attempt."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=3,
            base_seconds=1.0,
            multiplier=2.0,
        )

        # Then
        assert result == 4.0  # base_seconds * (2.0 ^ 2)

    def test_calculate_backoff_respects_max_seconds(self) -> None:
        """Test backoff is capped at max_seconds."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=10,
            base_seconds=1.0,
            multiplier=2.0,
            max_seconds=60.0,
        )

        # Then
        assert result == 60.0  # Capped at max_seconds

    def test_calculate_backoff_with_custom_multiplier(self) -> None:
        """Test backoff calculation with custom multiplier."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=2,
            base_seconds=2.0,
            multiplier=3.0,
        )

        # Then
        assert result == 6.0  # 2.0 * (3.0 ^ 1)

    def test_calculate_backoff_returns_zero_for_zero_attempt(self) -> None:
        """Test returns 0 for retry_attempt of 0 or less."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=0,
            base_seconds=1.0,
            multiplier=2.0,
        )

        # Then
        assert result == 0.0

    def test_calculate_backoff_returns_zero_for_negative_attempt(self) -> None:
        """Test returns 0 for negative retry_attempt."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=-1,
            base_seconds=1.0,
            multiplier=2.0,
        )

        # Then
        assert result == 0.0

    @pytest.mark.parametrize(
        "retry_attempt,base_seconds,multiplier,expected",
        [
            (1, 1.0, 2.0, 1.0),
            (2, 1.0, 2.0, 2.0),
            (3, 1.0, 2.0, 4.0),
            (4, 1.0, 2.0, 8.0),
            (1, 5.0, 1.5, 5.0),
            (2, 5.0, 1.5, 7.5),
            (3, 5.0, 1.5, 11.25),
        ],
    )
    def test_calculate_backoff_parametrized(
        self,
        retry_attempt: int,
        base_seconds: float,
        multiplier: float,
        expected: float,
    ) -> None:
        """Test backoff calculation with various parameter combinations."""
        # Given / When
        result = calculate_backoff_seconds(
            retry_attempt=retry_attempt,
            base_seconds=base_seconds,
            multiplier=multiplier,
        )

        # Then
        assert result == pytest.approx(expected)
