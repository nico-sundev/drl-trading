"""Unit tests for StatePersistenceService."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_preprocess.core.model.resample.resampling_context import ResamplingContext
from drl_trading_preprocess.adapter.resampling.state_persistence_service import StatePersistenceService


class TestStatePersistenceServiceInit:
    """Test StatePersistenceService initialization."""

    def test_init_creates_directory_if_not_exists(self) -> None:
        """Test that initialization creates state directory if it doesn't exist."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "non_existent" / "state.json"

            # When
            service = StatePersistenceService(str(state_path), backup_interval=500)

            # Then
            assert service.state_file_path == state_path
            assert service.backup_interval == 500
            assert service.operation_count == 0
            assert state_path.parent.exists()

    def test_init_with_existing_directory(self) -> None:
        """Test initialization with existing directory."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"

            # When
            service = StatePersistenceService(str(state_path))

            # Then
            assert service.state_file_path == state_path
            assert service.backup_interval == 1000  # Default value
            assert service.operation_count == 0


class TestStatePersistenceServiceSaveContext:
    """Test context saving functionality."""

    def test_save_context_success(self) -> None:
        """Test successful context saving."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path))

            context = ResamplingContext(max_symbols_in_memory=50)
            context.update_last_processed_timestamp(
                "BTCUSDT",
                Timeframe.MINUTE_1,
                datetime(2024, 1, 1, 10, 0, 0)
            )

            # When
            result = service.save_context(context)

            # Then
            assert result is True
            assert state_path.exists()

            # Verify saved content
            with open(state_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data['max_symbols_in_memory'] == 50
            assert 'BTCUSDT' in saved_data['symbol_states']
            assert '1m' in saved_data['symbol_states']['BTCUSDT']

    def test_save_context_creates_backup_of_existing_file(self) -> None:
        """Test that existing state file is backed up before saving new one."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            backup_path = Path(temp_dir) / "state.json.backup"

            # Create existing state file
            existing_data = {"test": "data"}
            with open(state_path, 'w') as f:
                json.dump(existing_data, f)

            service = StatePersistenceService(str(state_path))
            context = ResamplingContext(max_symbols_in_memory=25)

            # When
            result = service.save_context(context)

            # Then
            assert result is True
            assert backup_path.exists()

            # Verify backup contains original data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            assert backup_data == existing_data

    def test_save_context_handles_json_serialization_error(self) -> None:
        """Test error handling during JSON serialization."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path))

            # Create a context with problematic data
            context = Mock(spec=ResamplingContext)
            context.max_symbols_in_memory = 100
            context.symbol_states = {"invalid": "data that can't be serialized"}

            # When
            result = service.save_context(context)

            # Then
            assert result is False

    @patch("builtins.open", side_effect=PermissionError("Access denied"))
    def test_save_context_handles_file_permission_error(self, mock_open_call) -> None:
        """Test error handling when file cannot be written due to permissions."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path))
            context = ResamplingContext(max_symbols_in_memory=100)

            # When
            result = service.save_context(context)

            # Then
            assert result is False


class TestStatePersistenceServiceLoadContext:
    """Test context loading functionality."""

    def test_load_context_file_not_exists(self) -> None:
        """Test loading when no state file exists."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "nonexistent.json"
            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is None

    def test_load_context_success(self) -> None:
        """Test successful context loading."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"

            # Create valid state data
            state_data = {
                'saved_at': datetime.now().isoformat(),
                'max_symbols_in_memory': 75,
                'symbol_states': {
                    'BTCUSDT': {
                        '1m': {
                            'symbol': 'BTCUSDT',
                            'timeframe': '1m',
                            'last_processed_timestamp': '2024-01-01T10:00:00',
                            'records_processed': 100,
                            'candles_generated': 20,
                            'last_updated': '2024-01-01T10:05:00'
                        }
                    }
                }
            }

            with open(state_path, 'w') as f:
                json.dump(state_data, f)

            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is not None
            assert isinstance(result, ResamplingContext)
            assert result.max_symbols_in_memory == 75

            # Verify symbol state was restored
            timestamp = result.get_last_processed_timestamp("BTCUSDT", Timeframe.MINUTE_1)
            assert timestamp == datetime(2024, 1, 1, 10, 0, 0)

    def test_load_context_handles_corrupted_json(self) -> None:
        """Test error handling with corrupted JSON file."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"

            # Create corrupted JSON file
            with open(state_path, 'w') as f:
                f.write("{ invalid json content }")

            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is None

    def test_load_context_falls_back_to_backup(self) -> None:
        """Test fallback to backup file when main file is corrupted."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            backup_path = Path(temp_dir) / "state.json.backup"

            # Create corrupted main file
            with open(state_path, 'w') as f:
                f.write("{ invalid json }")

            # Create valid backup file
            backup_data = {
                'saved_at': datetime.now().isoformat(),
                'max_symbols_in_memory': 200,
                'symbol_states': {}
            }
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f)

            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is not None
            assert result.max_symbols_in_memory == 200

    def test_load_context_handles_missing_fields(self) -> None:
        """Test loading context with missing optional fields."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"

            # Create state data with missing optional fields
            state_data = {
                'symbol_states': {
                    'ETHUSDT': {
                        '5m': {
                            'symbol': 'ETHUSDT',
                            'timeframe': '5m',
                            # Missing optional fields
                        }
                    }
                }
            }

            with open(state_path, 'w') as f:
                json.dump(state_data, f)

            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is not None
            assert result.max_symbols_in_memory == 100  # Default value

            # Verify symbol state was created with defaults
            timestamp = result.get_last_processed_timestamp("ETHUSDT", Timeframe.MINUTE_5)
            assert timestamp is None

    def test_load_context_backup_also_corrupted(self) -> None:
        """Test behavior when both main file and backup file are corrupted."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            backup_path = Path(temp_dir) / "state.json.backup"

            # Create corrupted main file
            with open(state_path, 'w') as f:
                f.write("{ invalid json }")

            # Create corrupted backup file
            with open(backup_path, 'w') as f:
                f.write("{ also invalid json }")

            service = StatePersistenceService(str(state_path))

            # When
            result = service.load_context()

            # Then
            assert result is None




class TestStatePersistenceServiceAutoSave:
    """Test auto-save functionality."""

    def test_auto_save_if_needed_below_interval(self) -> None:
        """Test that auto-save doesn't trigger below interval."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path), backup_interval=5)
            context = ResamplingContext(max_symbols_in_memory=100)

            # When
            for _ in range(4):  # Below interval
                service.auto_save_if_needed(context)

            # Then
            assert not state_path.exists()
            assert service.operation_count == 4

    def test_auto_save_if_needed_at_interval(self) -> None:
        """Test that auto-save triggers at interval."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path), backup_interval=3)
            context = ResamplingContext(max_symbols_in_memory=100)

            # When
            for _ in range(3):  # At interval
                service.auto_save_if_needed(context)

            # Then
            assert state_path.exists()
            assert service.operation_count == 0  # Reset after save

    def test_auto_save_if_needed_multiple_intervals(self) -> None:
        """Test auto-save across multiple intervals."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path), backup_interval=2)
            context = ResamplingContext(max_symbols_in_memory=100)

            # When
            for _ in range(5):
                service.auto_save_if_needed(context)

            # Then
            assert state_path.exists()
            assert service.operation_count == 1  # 5 % 2 = 1


class TestStatePersistenceServiceCleanup:
    """Test cleanup functionality."""

    def test_cleanup_state_file_success(self) -> None:
        """Test successful cleanup of state files."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            backup_path = Path(temp_dir) / "state.json.backup"

            # Create both files
            state_path.touch()
            backup_path.touch()

            service = StatePersistenceService(str(state_path))

            # When
            result = service.cleanup_state_file()

            # Then
            assert result is True
            assert not state_path.exists()
            assert not backup_path.exists()

    def test_cleanup_state_file_only_main_file_exists(self) -> None:
        """Test cleanup when only main state file exists."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            state_path.touch()

            service = StatePersistenceService(str(state_path))

            # When
            result = service.cleanup_state_file()

            # Then
            assert result is True
            assert not state_path.exists()

    def test_cleanup_state_file_no_files_exist(self) -> None:
        """Test cleanup when no files exist."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "nonexistent.json"
            service = StatePersistenceService(str(state_path))

            # When
            result = service.cleanup_state_file()

            # Then
            assert result is True

    @patch("pathlib.Path.unlink", side_effect=PermissionError("Access denied"))
    def test_cleanup_state_file_permission_error(self, mock_unlink) -> None:
        """Test cleanup error handling with permission issues."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            state_path.touch()

            service = StatePersistenceService(str(state_path))

            # When
            result = service.cleanup_state_file()

            # Then
            assert result is False


class TestStatePersistenceServiceIntegration:
    """Integration tests for state persistence."""

    def test_save_and_load_round_trip(self) -> None:
        """Test complete save and load cycle."""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            service = StatePersistenceService(str(state_path))

            # Create context with complex state
            original_context = ResamplingContext(max_symbols_in_memory=150)
            original_context.update_last_processed_timestamp(
                "BTCUSDT",
                Timeframe.MINUTE_1,
                datetime(2024, 1, 1, 10, 0, 0)
            )
            original_context.update_last_processed_timestamp(
                "ETHUSDT",
                Timeframe.MINUTE_5,
                datetime(2024, 1, 1, 11, 0, 0)
            )
            original_context.increment_stats("BTCUSDT", Timeframe.MINUTE_1, 100, 20)
            original_context.increment_stats("ETHUSDT", Timeframe.MINUTE_5, 50, 10)

            # When
            save_result = service.save_context(original_context)
            loaded_context = service.load_context()

            # Then
            assert save_result is True
            assert loaded_context is not None
            assert loaded_context.max_symbols_in_memory == 150

            # Verify timestamps
            btc_timestamp = loaded_context.get_last_processed_timestamp("BTCUSDT", Timeframe.MINUTE_1)
            eth_timestamp = loaded_context.get_last_processed_timestamp("ETHUSDT", Timeframe.MINUTE_5)

            assert btc_timestamp == datetime(2024, 1, 1, 10, 0, 0)
            assert eth_timestamp == datetime(2024, 1, 1, 11, 0, 0)
