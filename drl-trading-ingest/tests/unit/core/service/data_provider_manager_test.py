"""
Unit tests for DataProviderManager.

Tests the manager's ability to register, retrieve, and manage data providers
in a thread-safe and consistent manner.
"""

from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from drl_trading_core.common.model.symbol_import_container import SymbolImportContainer
from drl_trading_ingest.core.port import DataProviderPort
from drl_trading_ingest.core.service.data_provider_manager import DataProviderManager


class MockDataProvider(DataProviderPort):
    """Mock data provider for testing."""

    def __init__(self, name: str = "mock_provider") -> None:
        super().__init__(config={})
        self._provider_name = name
        self.teardown_called = False

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name

    def setup(self) -> None:
        """Mock setup implementation."""
        self._is_initialized = True

    def teardown(self) -> None:
        """Mock teardown implementation."""
        self.teardown_called = True

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolImportContainer]:
        """Mock implementation."""
        return []

    def map_symbol(self, internal_symbol: str) -> str:
        """Mock implementation."""
        return internal_symbol

    def start_streaming(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Mock implementation."""
        pass

    def stop_streaming(self) -> None:
        """Mock implementation."""
        pass


class TestDataProviderManager:
    """Test suite for DataProviderManager core functionality."""

    @pytest.fixture
    def manager(self) -> DataProviderManager:
        """Create a fresh DataProviderManager instance for each test."""
        return DataProviderManager()

    @pytest.fixture
    def mock_provider(self) -> MockDataProvider:
        """Create a mock provider instance."""
        return MockDataProvider("test_provider")

    def test_initialization_creates_empty_registry(self, manager: DataProviderManager) -> None:
        """Test that manager initializes with an empty provider registry."""
        # Given / When
        # Manager is initialized via fixture

        # Then
        assert len(manager.get_all_providers()) == 0
        assert manager.get_available_provider_names() == []

    def test_register_provider_stores_provider_successfully(
        self, manager: DataProviderManager, mock_provider: MockDataProvider
    ) -> None:
        """Test that register_provider stores a provider with the given name."""
        # Given
        provider_name = "csv"

        # When
        manager.register_provider(provider_name, mock_provider)

        # Then
        assert manager.get_provider(provider_name) == mock_provider
        assert provider_name in manager.get_available_provider_names()
        assert len(manager.get_all_providers()) == 1

    def test_register_multiple_providers(self, manager: DataProviderManager) -> None:
        """Test registering multiple providers with different names."""
        # Given
        provider1 = MockDataProvider("provider1")
        provider2 = MockDataProvider("provider2")
        provider3 = MockDataProvider("provider3")

        # When
        manager.register_provider("csv", provider1)
        manager.register_provider("binance", provider2)
        manager.register_provider("yahoo", provider3)

        # Then
        assert len(manager.get_all_providers()) == 3
        assert manager.get_provider("csv") == provider1
        assert manager.get_provider("binance") == provider2
        assert manager.get_provider("yahoo") == provider3
        assert set(manager.get_available_provider_names()) == {"csv", "binance", "yahoo"}

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_register_provider_warns_on_override(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that registering a provider with an existing name logs a warning."""
        # Given
        provider1 = MockDataProvider("original")
        provider2 = MockDataProvider("replacement")
        provider_name = "csv"
        manager.register_provider(provider_name, provider1)

        # When
        manager.register_provider(provider_name, provider2)

        # Then
        mock_logger.warning.assert_called_once()
        assert "Overriding existing provider: csv" in str(mock_logger.warning.call_args)
        assert manager.get_provider(provider_name) == provider2

    def test_get_provider_returns_none_for_nonexistent_provider(
        self, manager: DataProviderManager
    ) -> None:
        """Test that get_provider returns None when provider is not found."""
        # Given
        nonexistent_name = "nonexistent_provider"

        # When
        result = manager.get_provider(nonexistent_name)

        # Then
        assert result is None

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_get_provider_logs_warning_when_not_found(
        self, mock_logger: Mock, manager: DataProviderManager, mock_provider: MockDataProvider
    ) -> None:
        """Test that get_provider logs a warning with available providers when not found."""
        # Given
        manager.register_provider("csv", mock_provider)
        nonexistent_name = "yahoo"

        # When
        manager.get_provider(nonexistent_name)

        # Then
        mock_logger.warning.assert_called_once()
        warning_message = str(mock_logger.warning.call_args)
        assert "yahoo" in warning_message
        assert "not found" in warning_message
        assert "csv" in warning_message

    def test_get_all_providers_returns_copy(
        self, manager: DataProviderManager, mock_provider: MockDataProvider
    ) -> None:
        """Test that get_all_providers returns a copy to prevent external mutation."""
        # Given
        manager.register_provider("csv", mock_provider)

        # When
        providers = manager.get_all_providers()
        providers["hacked"] = Mock()

        # Then
        assert "hacked" not in manager.get_all_providers()
        assert len(manager.get_all_providers()) == 1

    def test_get_all_providers_returns_all_registered_providers(
        self, manager: DataProviderManager
    ) -> None:
        """Test that get_all_providers returns all registered providers."""
        # Given
        provider1 = MockDataProvider("p1")
        provider2 = MockDataProvider("p2")
        manager.register_provider("csv", provider1)
        manager.register_provider("binance", provider2)

        # When
        all_providers = manager.get_all_providers()

        # Then
        assert len(all_providers) == 2
        assert all_providers["csv"] == provider1
        assert all_providers["binance"] == provider2

    def test_get_available_provider_names_returns_all_names(
        self, manager: DataProviderManager
    ) -> None:
        """Test that get_available_provider_names returns list of all provider names."""
        # Given
        manager.register_provider("csv", MockDataProvider())
        manager.register_provider("binance", MockDataProvider())
        manager.register_provider("yahoo", MockDataProvider())

        # When
        names = manager.get_available_provider_names()

        # Then
        assert len(names) == 3
        assert set(names) == {"csv", "binance", "yahoo"}

    def test_get_available_provider_names_returns_empty_list_when_no_providers(
        self, manager: DataProviderManager
    ) -> None:
        """Test that get_available_provider_names returns empty list when no providers registered."""
        # Given / When
        names = manager.get_available_provider_names()

        # Then
        assert names == []


class TestDataProviderManagerTeardown:
    """Test suite for DataProviderManager teardown functionality."""

    @pytest.fixture
    def manager(self) -> DataProviderManager:
        """Create a fresh DataProviderManager instance for each test."""
        return DataProviderManager()

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_teardown_all_calls_teardown_on_all_providers(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all calls teardown on all registered providers."""
        # Given
        provider1 = MockDataProvider("p1")
        provider2 = MockDataProvider("p2")
        provider3 = MockDataProvider("p3")
        manager.register_provider("csv", provider1)
        manager.register_provider("binance", provider2)
        manager.register_provider("yahoo", provider3)

        # When
        manager.teardown_all()

        # Then
        assert provider1.teardown_called is True
        assert provider2.teardown_called is True
        assert provider3.teardown_called is True

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_teardown_all_clears_provider_registry(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all clears the provider registry."""
        # Given
        manager.register_provider("csv", MockDataProvider())
        manager.register_provider("binance", MockDataProvider())
        assert len(manager.get_all_providers()) == 2

        # When
        manager.teardown_all()

        # Then
        assert len(manager.get_all_providers()) == 0
        assert manager.get_available_provider_names() == []

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_teardown_all_logs_info_message(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all logs an info message with provider count."""
        # Given
        manager.register_provider("csv", MockDataProvider())
        manager.register_provider("binance", MockDataProvider())

        # When
        manager.teardown_all()

        # Then
        mock_logger.info.assert_called_once()
        info_message = str(mock_logger.info.call_args)
        assert "Tearing down" in info_message
        assert "2 providers" in info_message

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_teardown_all_handles_provider_exceptions_gracefully(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all handles exceptions from provider teardown gracefully."""
        # Given
        faulty_provider = Mock()
        faulty_provider.teardown.side_effect = RuntimeError("Teardown failed")
        working_provider = MockDataProvider()

        manager.register_provider("faulty", faulty_provider)
        manager.register_provider("working", working_provider)

        # When
        manager.teardown_all()

        # Then
        mock_logger.error.assert_called_once()
        error_message = str(mock_logger.error.call_args)
        assert "faulty" in error_message
        assert "Teardown failed" in error_message
        assert working_provider.teardown_called is True
        assert len(manager.get_all_providers()) == 0

    @patch("drl_trading_ingest.core.service.data_provider_manager.logger")
    def test_teardown_all_logs_debug_for_successful_teardowns(
        self, mock_logger: Mock, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all logs debug messages for successful teardowns."""
        # Given
        manager.register_provider("csv", MockDataProvider())

        # When
        manager.teardown_all()

        # Then
        assert mock_logger.debug.call_count == 2  # register + teardown
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("Teardown completed" in call for call in debug_calls)

    def test_teardown_all_works_with_empty_registry(
        self, manager: DataProviderManager
    ) -> None:
        """Test that teardown_all works gracefully with no registered providers."""
        # Given / When
        manager.teardown_all()

        # Then
        assert len(manager.get_all_providers()) == 0


class TestDataProviderManagerEdgeCases:
    """Test suite for DataProviderManager edge cases and boundary conditions."""

    @pytest.fixture
    def manager(self) -> DataProviderManager:
        """Create a fresh DataProviderManager instance for each test."""
        return DataProviderManager()

    def test_register_provider_with_empty_name(
        self, manager: DataProviderManager, mock_provider: MockDataProvider = None
    ) -> None:
        """Test registering a provider with an empty string name."""
        # Given
        if mock_provider is None:
            mock_provider = MockDataProvider()
        empty_name = ""

        # When
        manager.register_provider(empty_name, mock_provider)

        # Then
        assert manager.get_provider(empty_name) == mock_provider

    def test_get_provider_with_empty_name(self, manager: DataProviderManager) -> None:
        """Test getting a provider with an empty string name."""
        # Given / When
        result = manager.get_provider("")

        # Then
        assert result is None

    def test_register_same_provider_instance_with_different_names(
        self, manager: DataProviderManager
    ) -> None:
        """Test registering the same provider instance under multiple names."""
        # Given
        provider = MockDataProvider()

        # When
        manager.register_provider("csv", provider)
        manager.register_provider("csv_backup", provider)

        # Then
        assert manager.get_provider("csv") is manager.get_provider("csv_backup")
        assert len(manager.get_all_providers()) == 2

    def test_teardown_all_multiple_times(self, manager: DataProviderManager) -> None:
        """Test that calling teardown_all multiple times is safe."""
        # Given
        manager.register_provider("csv", MockDataProvider())

        # When
        manager.teardown_all()
        manager.teardown_all()  # Second call should be safe

        # Then
        assert len(manager.get_all_providers()) == 0

    def test_operations_after_teardown(self, manager: DataProviderManager) -> None:
        """Test that manager can be used again after teardown_all."""
        # Given
        manager.register_provider("csv", MockDataProvider())
        manager.teardown_all()

        # When
        new_provider = MockDataProvider()
        manager.register_provider("binance", new_provider)

        # Then
        assert manager.get_provider("binance") == new_provider
        assert len(manager.get_all_providers()) == 1
