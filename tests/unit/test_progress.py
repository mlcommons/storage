"""
Tests for progress indication utilities in mlpstorage.progress module.

Tests cover:
- TTY detection
- progress_context in interactive/non-interactive modes
- create_stage_progress in interactive/non-interactive modes
- Cleanup on exceptions
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from mlpstorage.progress import (
    is_interactive_terminal,
    progress_context,
    create_stage_progress,
)


class TestIsInteractiveTerminal:
    """Tests for is_interactive_terminal function."""

    def test_returns_bool(self):
        """Should return a boolean value."""
        result = is_interactive_terminal()
        assert isinstance(result, bool)

    def test_returns_true_when_console_is_terminal(self):
        """Should return True when Console.is_terminal is True."""
        with patch("mlpstorage.progress.Console") as MockConsole:
            mock_console = MagicMock()
            type(mock_console).is_terminal = PropertyMock(return_value=True)
            MockConsole.return_value = mock_console

            result = is_interactive_terminal()

        assert result is True

    def test_returns_false_when_console_is_not_terminal(self):
        """Should return False when Console.is_terminal is False."""
        with patch("mlpstorage.progress.Console") as MockConsole:
            mock_console = MagicMock()
            type(mock_console).is_terminal = PropertyMock(return_value=False)
            MockConsole.return_value = mock_console

            result = is_interactive_terminal()

        assert result is False


class TestProgressContextNonInteractive:
    """Tests for progress_context in non-interactive mode."""

    def test_logs_status_with_logger(self):
        """Should log status via logger.status() in non-interactive mode."""
        mock_logger = MagicMock()

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with progress_context("Loading data", logger=mock_logger) as (
                update,
                set_desc,
            ):
                pass

        mock_logger.status.assert_called_once_with("Loading data...")

    def test_no_error_without_logger(self):
        """Should not error when no logger is provided in non-interactive mode."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with progress_context("Loading data") as (update, set_desc):
                # Should not raise any exceptions
                pass

    def test_yielded_functions_are_noops(self):
        """Should yield no-op functions that can be called without error."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with progress_context("Loading data") as (update, set_desc):
                # These should not raise any exceptions
                update()
                update(advance=5)
                update(completed=50)
                set_desc("New description")


class TestProgressContextInteractive:
    """Tests for progress_context in interactive mode."""

    def test_creates_progress_for_indeterminate(self):
        """Should create Progress with spinner for indeterminate (total=None)."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with progress_context("Loading") as (update, set_desc):
                    pass

                # Verify Progress was created and started/stopped
                MockProgress.assert_called_once()
                mock_progress.start.assert_called_once()
                mock_progress.stop.assert_called_once()

    def test_creates_progress_for_determinate(self):
        """Should create Progress with bar for determinate (total set)."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with progress_context("Processing", total=100) as (update, set_desc):
                    pass

                # Verify Progress was created and add_task called with total
                mock_progress.add_task.assert_called_once()
                call_kwargs = mock_progress.add_task.call_args
                assert call_kwargs[1]["total"] == 100

    def test_update_advances_progress(self):
        """Should advance progress when update is called."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with progress_context("Processing", total=100) as (update, set_desc):
                    update(advance=5)

                # Verify progress.update was called with advance=5
                mock_progress.update.assert_called()
                # Check the call was made with advance=5
                calls = mock_progress.update.call_args_list
                assert any(
                    call[1].get("advance") == 5 for call in calls
                ), f"Expected advance=5 in calls: {calls}"

    def test_update_sets_completed(self):
        """Should set completed value when update is called with completed."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with progress_context("Processing", total=100) as (update, set_desc):
                    update(completed=50)

                # Verify progress.update was called with completed=50
                calls = mock_progress.update.call_args_list
                assert any(
                    call[1].get("completed") == 50 for call in calls
                ), f"Expected completed=50 in calls: {calls}"

    def test_set_description_updates(self):
        """Should update description when set_description is called."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with progress_context("Processing") as (update, set_desc):
                    set_desc("New description")

                # Verify progress.update was called with description
                calls = mock_progress.update.call_args_list
                assert any(
                    call[1].get("description") == "New description" for call in calls
                ), f"Expected description='New description' in calls: {calls}"

    def test_exception_cleanup(self):
        """Should stop progress even when exception is raised inside context."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with pytest.raises(ValueError):
                    with progress_context("Processing") as (update, set_desc):
                        raise ValueError("Test exception")

                # Verify progress.stop was still called
                mock_progress.stop.assert_called_once()


class TestCreateStageProgressNonInteractive:
    """Tests for create_stage_progress in non-interactive mode."""

    def test_logs_stages_with_logger(self):
        """Should log each stage via logger.status() in non-interactive mode."""
        mock_logger = MagicMock()
        stages = ["Stage 1", "Stage 2", "Stage 3"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with create_stage_progress(stages, logger=mock_logger) as advance_stage:
                # Initial stage already logged
                advance_stage()  # Advance to Stage 2
                advance_stage()  # Advance to Stage 3

        # Verify logger.status was called for all stages
        assert mock_logger.status.call_count == 3
        calls = [str(call) for call in mock_logger.status.call_args_list]
        assert any("Stage 1" in call for call in calls)
        assert any("Stage 2" in call for call in calls)
        assert any("Stage 3" in call for call in calls)

    def test_no_error_without_logger(self):
        """Should not error when no logger is provided in non-interactive mode."""
        stages = ["Stage 1", "Stage 2"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with create_stage_progress(stages) as advance_stage:
                advance_stage()  # Should not raise

    def test_empty_stages_works(self):
        """Should handle empty stages list without error."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=False
        ):
            with create_stage_progress([]) as advance_stage:
                advance_stage()  # Should not raise


class TestCreateStageProgressInteractive:
    """Tests for create_stage_progress in interactive mode."""

    def test_creates_progress_with_total_stages(self):
        """Should create Progress with total=len(stages)."""
        stages = ["Validating", "Collecting", "Running"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with create_stage_progress(stages) as advance_stage:
                    pass

                # Verify Progress was created
                MockProgress.assert_called_once()
                # Verify add_task was called with total=3
                mock_progress.add_task.assert_called_once()
                call_kwargs = mock_progress.add_task.call_args
                assert call_kwargs[1]["total"] == 3

    def test_advance_stage_updates_progress(self):
        """Should update progress when advance_stage is called."""
        stages = ["Stage 1", "Stage 2", "Stage 3"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with create_stage_progress(stages) as advance_stage:
                    advance_stage()  # Advance to Stage 2

                # Verify progress.update was called
                mock_progress.update.assert_called()
                calls = mock_progress.update.call_args_list
                assert any(
                    call[1].get("advance") == 1 for call in calls
                ), f"Expected advance=1 in calls: {calls}"

    def test_advance_stage_with_custom_name(self):
        """Should update description when advance_stage is called with stage_name."""
        stages = ["Stage 1", "Stage 2"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with create_stage_progress(stages) as advance_stage:
                    advance_stage("Custom Stage Name")

                # Verify progress.update was called with custom description
                calls = mock_progress.update.call_args_list
                assert any(
                    call[1].get("description") == "Custom Stage Name" for call in calls
                ), f"Expected description='Custom Stage Name' in calls: {calls}"

    def test_exception_cleanup(self):
        """Should stop progress even when exception is raised."""
        stages = ["Stage 1", "Stage 2"]

        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                mock_progress = MagicMock()
                mock_progress.add_task.return_value = 0
                MockProgress.return_value = mock_progress

                with pytest.raises(ValueError):
                    with create_stage_progress(stages) as advance_stage:
                        raise ValueError("Test exception")

                # Verify progress.stop was still called
                mock_progress.stop.assert_called_once()

    def test_empty_stages_interactive(self):
        """Should handle empty stages list without creating Progress."""
        with patch(
            "mlpstorage.progress.is_interactive_terminal", return_value=True
        ):
            with patch("mlpstorage.progress.Progress") as MockProgress:
                with create_stage_progress([]) as advance_stage:
                    advance_stage()  # Should not raise

                # Progress should NOT be created for empty stages
                MockProgress.assert_not_called()
