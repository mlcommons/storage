"""
Tests for HistoryTracker class in mlpstorage.history module.

Tests cover:
- History file creation and management
- Adding and retrieving history entries
- Parsing history lines
- Sequence ID generation
- Command retrieval by ID
- History printing
- Args recreation from history
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from mlpstorage.history import HistoryTracker
from mlpstorage.config import EXIT_CODE


class TestHistoryTrackerInit:
    """Tests for HistoryTracker initialization."""

    def test_creates_history_file_if_not_exists(self, tmp_path):
        """Should create history file if it doesn't exist."""
        history_file = tmp_path / "subdir" / "history.txt"
        tracker = HistoryTracker(history_file=str(history_file))
        assert history_file.exists()

    def test_uses_default_history_file(self):
        """Should use default history file path."""
        with patch('mlpstorage.history.HISTFILE', '/tmp/test_history'):
            with patch('os.path.exists', return_value=True):
                tracker = HistoryTracker()
                assert tracker.history_file == '/tmp/test_history'

    def test_accepts_custom_logger(self, tmp_path):
        """Should accept custom logger."""
        history_file = tmp_path / "history.txt"
        mock_logger = MagicMock()
        tracker = HistoryTracker(history_file=str(history_file), logger=mock_logger)
        assert tracker.logger == mock_logger


class TestParseHistoryLine:
    """Tests for _parse_history_line method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker instance."""
        history_file = tmp_path / "history.txt"
        return HistoryTracker(history_file=str(history_file))

    def test_parses_valid_line(self, tracker):
        """Should parse valid history line."""
        line = "1,20250111_143022,mlpstorage training run --model unet3d"
        result = tracker._parse_history_line(line)
        assert result == (1, "20250111_143022", "mlpstorage training run --model unet3d")

    def test_parses_line_with_commas_in_command(self, tracker):
        """Should handle commas in command."""
        line = "2,20250111_150000,mlpstorage training run --hosts host1,host2"
        result = tracker._parse_history_line(line)
        assert result == (2, "20250111_150000", "mlpstorage training run --hosts host1,host2")

    def test_returns_none_for_invalid_line(self, tracker):
        """Should return None for invalid line."""
        line = "invalid line without proper format"
        result = tracker._parse_history_line(line)
        assert result is None

    def test_returns_none_for_empty_line(self, tracker):
        """Should return None for empty line."""
        line = ""
        result = tracker._parse_history_line(line)
        assert result is None


class TestGetNextSequenceId:
    """Tests for get_next_sequence_id method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker instance."""
        history_file = tmp_path / "history.txt"
        return HistoryTracker(history_file=str(history_file))

    def test_returns_1_for_empty_file(self, tracker):
        """Should return 1 for empty history file."""
        assert tracker.get_next_sequence_id() == 1

    def test_returns_incremented_id(self, tracker):
        """Should return last ID + 1."""
        with open(tracker.history_file, 'w') as f:
            f.write("1,20250111_143022,command1\n")
            f.write("2,20250111_144000,command2\n")
            f.write("3,20250111_145000,command3\n")
        assert tracker.get_next_sequence_id() == 4

    def test_handles_nonexistent_file(self, tmp_path):
        """Should return 1 for nonexistent file."""
        history_file = tmp_path / "nonexistent.txt"
        tracker = HistoryTracker.__new__(HistoryTracker)
        tracker.history_file = str(history_file)
        tracker.logger = MagicMock()
        assert tracker.get_next_sequence_id() == 1


class TestAddEntry:
    """Tests for add_entry method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker instance."""
        history_file = tmp_path / "history.txt"
        return HistoryTracker(history_file=str(history_file))

    def test_adds_string_command(self, tracker):
        """Should add string command to history."""
        seq_id = tracker.add_entry("mlpstorage training run --model unet3d")
        assert seq_id == 1
        with open(tracker.history_file, 'r') as f:
            content = f.read()
        assert "mlpstorage training run --model unet3d" in content

    def test_adds_list_command(self, tracker):
        """Should join list command to string."""
        seq_id = tracker.add_entry(["mlpstorage", "training", "run", "--model", "unet3d"])
        assert seq_id == 1
        with open(tracker.history_file, 'r') as f:
            content = f.read()
        assert "mlpstorage training run --model unet3d" in content

    def test_uses_custom_datetime(self, tracker):
        """Should use custom datetime string."""
        seq_id = tracker.add_entry("command", datetime_str="20250115_120000")
        with open(tracker.history_file, 'r') as f:
            content = f.read()
        assert "20250115_120000" in content

    def test_increments_sequence_id(self, tracker):
        """Should increment sequence ID for each entry."""
        seq1 = tracker.add_entry("command1")
        seq2 = tracker.add_entry("command2")
        seq3 = tracker.add_entry("command3")
        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3


class TestGetCommandById:
    """Tests for get_command_by_id method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker with entries."""
        history_file = tmp_path / "history.txt"
        history_file.write_text(
            "1,20250111_143022,mlpstorage training run --model unet3d\n"
            "2,20250111_144000,mlpstorage checkpointing run --model llama3-8b\n"
            "3,20250111_145000,mlpstorage reports reportgen\n"
        )
        return HistoryTracker(history_file=str(history_file))

    def test_retrieves_existing_command(self, tracker):
        """Should retrieve command by ID."""
        command = tracker.get_command_by_id(2)
        assert command == "mlpstorage checkpointing run --model llama3-8b"

    def test_returns_none_for_nonexistent_id(self, tracker):
        """Should return None for nonexistent ID."""
        command = tracker.get_command_by_id(999)
        assert command is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Should return None for empty file."""
        history_file = tmp_path / "empty.txt"
        tracker = HistoryTracker(history_file=str(history_file))
        command = tracker.get_command_by_id(1)
        assert command is None


class TestGetHistoryEntries:
    """Tests for get_history_entries method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker with entries."""
        history_file = tmp_path / "history.txt"
        history_file.write_text(
            "1,20250111_143022,command1\n"
            "2,20250111_144000,command2\n"
            "3,20250111_145000,command3\n"
            "4,20250111_146000,command4\n"
            "5,20250111_147000,command5\n"
        )
        return HistoryTracker(history_file=str(history_file))

    def test_returns_all_entries(self, tracker):
        """Should return all entries when no limit."""
        entries = tracker.get_history_entries()
        assert len(entries) == 5
        assert entries[0] == (1, "20250111_143022", "command1")

    def test_returns_limited_entries(self, tracker):
        """Should return limited entries from end."""
        entries = tracker.get_history_entries(limit=2)
        assert len(entries) == 2
        assert entries[0] == (4, "20250111_146000", "command4")
        assert entries[1] == (5, "20250111_147000", "command5")

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        """Should return empty list for empty file."""
        history_file = tmp_path / "empty.txt"
        tracker = HistoryTracker(history_file=str(history_file))
        entries = tracker.get_history_entries()
        assert entries == []

    def test_handles_zero_limit(self, tracker):
        """Should return all entries for limit=0."""
        entries = tracker.get_history_entries(limit=0)
        assert len(entries) == 5


class TestPrintHistory:
    """Tests for print_history method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker with entries."""
        history_file = tmp_path / "history.txt"
        history_file.write_text(
            "1,20250111_143022,command1\n"
            "2,20250111_144000,command2\n"
        )
        return HistoryTracker(history_file=str(history_file))

    def test_prints_all_entries(self, tracker, capsys):
        """Should print all entries."""
        result = tracker.print_history()
        captured = capsys.readouterr()
        assert "command1" in captured.out
        assert "command2" in captured.out
        assert result == EXIT_CODE.SUCCESS

    def test_prints_specific_entry(self, tracker, capsys):
        """Should print specific entry by ID."""
        result = tracker.print_history(sequence_id=1)
        captured = capsys.readouterr()
        assert "command1" in captured.out
        assert "command2" not in captured.out
        assert result == EXIT_CODE.SUCCESS

    def test_prints_limited_entries(self, tracker, capsys):
        """Should print limited entries."""
        result = tracker.print_history(limit=1)
        captured = capsys.readouterr()
        assert "command2" in captured.out
        assert result == EXIT_CODE.SUCCESS

    def test_returns_error_for_nonexistent_id(self, tracker, capsys):
        """Should return error for nonexistent ID."""
        result = tracker.print_history(sequence_id=999)
        captured = capsys.readouterr()
        assert "no command found" in captured.out.lower()
        assert result == EXIT_CODE.INVALID_ARGUMENTS

    def test_returns_error_for_empty_file(self, tmp_path, capsys):
        """Should return error for empty file."""
        history_file = tmp_path / "empty.txt"
        tracker = HistoryTracker(history_file=str(history_file))
        result = tracker.print_history()
        captured = capsys.readouterr()
        assert "no history" in captured.out.lower()
        assert result == EXIT_CODE.INVALID_ARGUMENTS


class TestHandleHistoryCommand:
    """Tests for handle_history_command method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker with entries."""
        history_file = tmp_path / "history.txt"
        history_file.write_text(
            "1,20250111_143022,mlpstorage training run --model unet3d\n"
        )
        return HistoryTracker(history_file=str(history_file))

    def test_handles_id_argument(self, tracker, capsys):
        """Should handle --id argument."""
        args = Namespace(id=1, limit=None, rerun_id=None)
        result = tracker.handle_history_command(args)
        captured = capsys.readouterr()
        assert "unet3d" in captured.out
        assert result == EXIT_CODE.SUCCESS

    def test_handles_limit_argument(self, tracker, capsys):
        """Should handle --limit argument."""
        args = Namespace(id=None, limit=1, rerun_id=None)
        result = tracker.handle_history_command(args)
        assert result == EXIT_CODE.SUCCESS

    def test_handles_no_arguments(self, tracker, capsys):
        """Should print all history when no arguments."""
        args = Namespace(id=None, limit=None, rerun_id=None)
        result = tracker.handle_history_command(args)
        assert result == EXIT_CODE.SUCCESS

    def test_handles_rerun_nonexistent_id(self, tracker, capsys):
        """Should return error for nonexistent rerun ID."""
        args = Namespace(id=None, limit=None, rerun_id=999)
        result = tracker.handle_history_command(args)
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()
        assert result == EXIT_CODE.INVALID_ARGUMENTS


class TestCreateArgsFromCommand:
    """Tests for create_args_from_command method."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a HistoryTracker with entries."""
        history_file = tmp_path / "history.txt"
        return HistoryTracker(history_file=str(history_file))

    def test_returns_none_for_nonexistent_command(self, tracker):
        """Should return None for nonexistent command."""
        result = tracker.create_args_from_command(999)
        assert result is None

    def test_removes_script_name_from_command(self, tracker):
        """Should remove script name from command parts."""
        # Add a command with script name
        tracker.add_entry("mlpstorage training datasize --model unet3d --max-accelerators 8 --accelerator-type h100 --client-host-memory-in-gb 128")

        # Mock parse_arguments to capture what's passed
        with patch('mlpstorage.cli.parse_arguments') as mock_parse:
            mock_parse.return_value = Namespace(program='training')
            tracker.create_args_from_command(1)
            # Verify sys.argv was set without the script name at front
            # (This is harder to test directly, so we verify parse_arguments was called)
            mock_parse.assert_called_once()


class TestHistoryTrackerIntegration:
    """Integration tests for HistoryTracker."""

    def test_full_workflow(self, tmp_path):
        """Test complete add-retrieve workflow."""
        history_file = tmp_path / "history.txt"
        tracker = HistoryTracker(history_file=str(history_file))

        # Add entries
        tracker.add_entry("mlpstorage training run --model unet3d", datetime_str="20250111_143022")
        tracker.add_entry("mlpstorage checkpointing run --model llama3-8b", datetime_str="20250111_144000")

        # Retrieve by ID
        cmd1 = tracker.get_command_by_id(1)
        cmd2 = tracker.get_command_by_id(2)
        assert "unet3d" in cmd1
        assert "llama3-8b" in cmd2

        # Get all entries
        entries = tracker.get_history_entries()
        assert len(entries) == 2

        # Get limited entries
        recent = tracker.get_history_entries(limit=1)
        assert len(recent) == 1
        assert "llama3-8b" in recent[0][2]

    def test_persistence_across_instances(self, tmp_path):
        """History should persist across tracker instances."""
        history_file = tmp_path / "history.txt"

        # First instance adds entries
        tracker1 = HistoryTracker(history_file=str(history_file))
        tracker1.add_entry("command1")
        tracker1.add_entry("command2")

        # Second instance should see entries
        tracker2 = HistoryTracker(history_file=str(history_file))
        entries = tracker2.get_history_entries()
        assert len(entries) == 2

        # Third entry should have correct sequence ID
        seq_id = tracker2.add_entry("command3")
        assert seq_id == 3
