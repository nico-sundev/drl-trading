"""
Conftest for resample service unit tests.

Provides common fixtures for testing the resampling service components.
"""
from datetime import datetime

import pytest

from drl_trading_preprocess.application.config.preprocess_config import (
    ResampleConfig,
)


@pytest.fixture
def resample_config() -> ResampleConfig:
    """Create a complete resample configuration for testing."""
    return ResampleConfig(
        historical_start_date=datetime(2020, 1, 1),
        max_batch_size=1000,
        progress_log_interval=1000,
        enable_incomplete_candle_publishing=True,
        chunk_size=500,
        memory_warning_threshold_mb=100,
        pagination_limit=10000,
        max_memory_usage_mb=500,
        state_persistence_enabled=False,
        state_file_path="/tmp/test_state.json",
        state_backup_interval=3600,
        auto_cleanup_inactive_symbols=True,
        inactive_symbol_threshold_hours=24,
    )
