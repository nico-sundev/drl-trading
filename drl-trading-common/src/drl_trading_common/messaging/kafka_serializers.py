"""JSON serialization utilities for Kafka messages.

This module provides serialization and deserialization functions for Kafka
messages using JSON format. It handles encoding/decoding and provides
type-safe interfaces for message payloads.
"""

import json
from typing import Any, Dict


def serialize_to_json(data: Dict[str, Any]) -> bytes:
    """Serialize a dictionary to JSON bytes for Kafka publishing.

    Args:
        data: Dictionary containing the message payload.

    Returns:
        UTF-8 encoded JSON bytes ready for Kafka.

    Raises:
        TypeError: If the data contains non-serializable types.
        ValueError: If the data structure is invalid.
    """
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def deserialize_from_json(data: bytes) -> Dict[str, Any]:
    """Deserialize JSON bytes from Kafka to a dictionary.

    Args:
        data: UTF-8 encoded JSON bytes from Kafka message.

    Returns:
        Dictionary containing the deserialized message payload.

    Raises:
        json.JSONDecodeError: If the data is not valid JSON.
        UnicodeDecodeError: If the data is not valid UTF-8.
    """
    result = json.loads(data.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict but got {type(result).__name__}")
    return result
