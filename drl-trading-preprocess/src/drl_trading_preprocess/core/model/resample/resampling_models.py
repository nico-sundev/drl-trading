"""
Data models for resampling service operations.

This module provides compatibility imports for the resampling data models
that have been moved to individual files for better separation of concerns.
"""

# Compatibility imports - prefer importing directly from specific modules
from drl_trading_preprocess.core.model.resample.resampling_request import ResamplingRequest
from drl_trading_preprocess.core.model.resample.resampling_response import ResamplingResponse
from drl_trading_preprocess.core.model.resample.resampling_error import ResamplingError
from drl_trading_preprocess.core.model.resample.timeframe_candle_accumulator import TimeframeCandleAccumulator

__all__ = [
    "ResamplingRequest",
    "ResamplingResponse",
    "ResamplingError",
    "TimeframeCandleAccumulator"
]
