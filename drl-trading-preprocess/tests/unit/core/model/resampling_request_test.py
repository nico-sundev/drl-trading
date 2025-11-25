"""
Tests for ResamplingRequest domain model.

This module contains unit tests for the resampling request validation logic,
including symbol validation, timeframe hierarchy validation, and request structure validation.
"""

import pytest

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.resample.resampling_request import ResamplingRequest


class TestResamplingRequest:
    """Test suite for ResamplingRequest validation."""

    def test_valid_request_creation(self):
        """Test creation of valid resampling request."""
        # Given
        symbol = "EURUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5, Timeframe.HOUR_1]

        # When
        request = ResamplingRequest(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
        )

        # Then
        assert request.symbol == symbol
        assert request.base_timeframe == base_timeframe
        assert request.target_timeframes == target_timeframes

    def test_invalid_symbol_validation(self):
        """Test validation of invalid symbols."""
        # Given
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]

        # When & Then
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            ResamplingRequest(
                symbol="",
                base_timeframe=base_timeframe,
                target_timeframes=target_timeframes,
            )

    def test_invalid_timeframe_hierarchy_validation(self):
        """Test validation of timeframe hierarchy."""
        # Given
        symbol = "EURUSD"
        base_timeframe = Timeframe.HOUR_1
        target_timeframes = [Timeframe.MINUTE_5]  # Smaller than base

        # When & Then
        with pytest.raises(
            ValueError, match="Target timeframe .* must be larger than base timeframe"
        ):
            ResamplingRequest(
                symbol=symbol,
                base_timeframe=base_timeframe,
                target_timeframes=target_timeframes,
            )

    def test_empty_target_timeframes_validation(self):
        """Test validation of empty target timeframes."""
        # Given
        symbol = "EURUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = []

        # When & Then
        with pytest.raises(ValueError, match="Target timeframes cannot be empty"):
            ResamplingRequest(
                symbol=symbol,
                base_timeframe=base_timeframe,
                target_timeframes=target_timeframes,
            )

    def test_multiple_target_timeframes(self):
        """Test request with multiple valid target timeframes."""
        # Given
        symbol = "BTCUSD"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1, Timeframe.HOUR_4]

        # When
        request = ResamplingRequest(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
        )

        # Then
        assert len(request.target_timeframes) == 4
        assert Timeframe.MINUTE_5 in request.target_timeframes
        assert Timeframe.HOUR_4 in request.target_timeframes

    def test_optional_timestamps(self):
        """Test request creation with optional timestamp parameters."""
        # Given
        from datetime import datetime
        symbol = "ETHUSDT"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5]
        start_timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # When
        request = ResamplingRequest(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            start_from_timestamp=start_timestamp
        )

        # Then
        assert request.start_from_timestamp == start_timestamp
        assert request.symbol == symbol
        assert request.base_timeframe == base_timeframe
        assert request.target_timeframes == target_timeframes

    def test_timeframe_validation_edge_cases(self):
        """Test edge cases in timeframe hierarchy validation."""
        # Given
        symbol = "EURUSD"

        # Test case 1: Same timeframe (should be invalid)
        with pytest.raises(ValueError):
            ResamplingRequest(
                symbol=symbol,
                base_timeframe=Timeframe.MINUTE_5,
                target_timeframes=[Timeframe.MINUTE_5],
            )

        # Test case 2: Mixed valid and invalid timeframes (should be invalid)
        with pytest.raises(ValueError):
            ResamplingRequest(
                symbol=symbol,
                base_timeframe=Timeframe.HOUR_1,
                target_timeframes=[Timeframe.HOUR_4, Timeframe.MINUTE_5],  # One valid, one invalid
            )

        # Test case 3: Multiple valid ascending timeframes (should be valid)
        request = ResamplingRequest(
            symbol=symbol,
            base_timeframe=Timeframe.MINUTE_1,
            target_timeframes=[Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1],
        )
        assert len(request.target_timeframes) == 3
