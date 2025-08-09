"""
Shared test fixtures for drl-trading-common tests.

Provides common fixtures for logging, configuration, and test utilities
following the Given/When/Then structure requirements.
"""

import logging
import os
import tempfile
import time
from typing import Generator
from unittest.mock import Mock
import pytest

from drl_trading_common.logging.trading_log_context import TradingLogContext
from drl_trading_common.model.trading_context import TradingContext
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """
    Provide a temporary directory for test files.

    Yields:
        Absolute path to temporary directory
    """
    import logging

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(original_cwd)
            # Close all logging handlers to release file handles
            logging.shutdown()
            # Small delay to ensure files are released
            time.sleep(0.1)


@pytest.fixture
def clean_log_context() -> Generator[None, None, None]:
    """
    Ensure clean logging context for each test.

    Clears TradingLogContext before and after test execution.
    """
    TradingLogContext.clear()
    try:
        yield
    finally:
        TradingLogContext.clear()


@pytest.fixture
def sample_trading_context() -> TradingContext:
    """
    Provide a sample trading context for testing.

    Returns:
        TradingContext with sample trading data
    """
    return TradingContext.create_initial_context(
        symbol="EURUSD",
        timeframe="1H",
        filename="test_data.csv"
    )


@pytest.fixture
def mock_logger() -> Mock:
    """
    Provide a mock logger for testing.

    Returns:
        Mock object configured as a logger
    """
    mock = Mock(spec=logging.Logger)
    mock.handlers = []
    mock.level = logging.INFO
    return mock


@pytest.fixture
def service_logger_config() -> ServiceLoggingConfig:
    """
    Provide standard ServiceLoggingConfig for testing.

    Returns:
        ServiceLoggingConfig with test-appropriate settings
    """
    return ServiceLoggingConfig(
        level='INFO',
        file_logging=True,
        max_file_size=1024 * 1024,  # 1MB for tests
        backup_count=3,
        console_enabled=True,
        third_party_loggers={
            'urllib3': {'level': 'WARNING'},
            'requests': {'level': 'WARNING'}
        }
    )


@pytest.fixture(params=['local', 'cicd', 'prod'])
def stage_environment(request) -> Generator[str, None, None]:
    """
    Parametrized fixture for different stage environments.

    Yields:
        Stage name and sets STAGE environment variable
    """
    original_env = os.environ.get('STAGE')
    os.environ['STAGE'] = request.param
    try:
        yield request.param
    finally:
        if original_env is not None:
            os.environ['STAGE'] = original_env
        elif 'STAGE' in os.environ:
            del os.environ['STAGE']
