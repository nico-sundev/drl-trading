"""Fixtures specific to preprocessing tests."""

from unittest.mock import MagicMock

import pytest
from drl_trading_common.base.technical_metrics_service_interface import (
    TechnicalMetricsServiceInterface,
)


@pytest.fixture
def mock_technical_metrics_service() -> TechnicalMetricsServiceInterface:
    """Create a mock technical metrics service for preprocessing tests."""
    service = MagicMock(spec=TechnicalMetricsServiceInterface)
    service.calculate_rsi.return_value = [50.0] * 10
    service.calculate_sma.return_value = [1.0] * 10
    service.calculate_ema.return_value = [1.0] * 10
    return service
