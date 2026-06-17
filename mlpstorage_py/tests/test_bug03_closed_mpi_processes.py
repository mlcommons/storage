"""Tests for BUG-03: closed_mpi_processes subset branch raises NameError.

In the CURRENT (buggy) code, ``closed_mpi_processes`` only assigns
``model_key`` inside the ``else:`` branch (line 160 of checkpointing_checks.py).
When ``checkpoint_mode == "subset"`` the ``if`` branch runs and references
``model_key`` at line 152 BEFORE it is defined, raising:

    NameError: name 'model_key' is not defined

The fix: derive ``model_key`` BEFORE the ``if checkpoint_mode == "subset"``
branch so it is always defined when used.

Additionally, the BUG-03 fix lifts the inline ``model_process_requirements``
dict to ``constants.CLOSED_MPI_PROCESSES`` and delegates via
``self.config.get_closed_mpi_processes(model_key)`` (D-C4).

The method is also upgraded to emit violations via ``self.log_violation``
with rule ID ``[4.6.1 checkpointClosedMpiProcesses]`` rather than bare
``self.log.error`` (QUAL-02 retrofit for the touched method).

References:
  - BUG-03 in REQUIREMENTS.md
  - D-C4 in Phase 2 CONTEXT.md
  - RESEARCH.md §Codebase Investigation: checkpointing_checks.py §BUG-03 site
"""

import pytest
from unittest.mock import MagicMock

from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import SubmissionLogs, LoaderMetadata

# Import conftest's MockLogger (importable because conftest is in the tests package)
from mlpstorage_py.tests.conftest import MockLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_check(
    *,
    verification: str = "closed",
    checkpoint_mode: str = "subset",
    model: str = "llama3-8b",
    num_processes: int = 7,
    mock_logger: MockLogger | None = None,
) -> CheckpointingCheck:
    """Construct a CheckpointingCheck with a single fake checkpoint entry.

    ``SubmissionLogs.checkpoint_files`` is set to a list with one
    ``(summary, metadata, timestamp)`` triple.  ``system_file`` is set to
    ``{}`` so D-D3's attribute exists without failing.
    """
    if mock_logger is None:
        mock_logger = MockLogger()

    metadata = {
        "verification": verification,
        "params_dict": {"checkpoint.mode": checkpoint_mode},
        "args": {
            "model": model,
            "num_processes": num_processes,
        },
    }
    summary = {}

    loader_metadata = LoaderMetadata(
        division="closed",
        submitter="Acme",
        system="sys-v1",
        mode="checkpointing",
        benchmark="llama3-8b",
        folder="/fake/path",
    )

    # Build a SubmissionLogs that holds the single entry
    sub_logs = SubmissionLogs(
        checkpoint_files=[(summary, metadata, "20250101_120000")],
        system_file={},
        loader_metadata=loader_metadata,
    )

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    return CheckpointingCheck(log=mock_logger, config=config, submissions_logs=sub_logs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBug03SubsetBranch:
    """BUG-03: subset branch must not raise NameError."""

    def test_bug03_subset_branch_does_not_raise_nameerror(self):
        """num_processes != 8 in subset mode must return False + violation, not NameError.

        BEFORE the fix: raises NameError because model_key is used before assignment.
        AFTER the fix: returns False AND mock_logger.errors contains a
        [4.6.1 checkpointClosedMpiProcesses] violation.
        """
        mock_log = MockLogger()
        check = _make_check(
            checkpoint_mode="subset",
            model="llama3-8b",
            num_processes=7,
            mock_logger=mock_log,
        )
        # Before the fix this raises NameError.
        result = check.closed_mpi_processes()
        assert result is False, "subset mode with wrong count should return False"
        assert any(
            "[4.6.1 checkpointClosedMpiProcesses]" in e for e in mock_log.errors
        ), (
            "Expected a [4.6.1 checkpointClosedMpiProcesses] violation. "
            f"Got errors: {mock_log.errors}"
        )
        # Message should mention "subset mode requires 8 processes" and the actual count
        assert any("subset" in e and "7" in e for e in mock_log.errors), (
            f"Violation message should mention 'subset' and '7'. Got: {mock_log.errors}"
        )

    def test_bug03_subset_branch_passes_for_correct_processes(self):
        """num_processes == 8 in subset mode must return True with no violations."""
        mock_log = MockLogger()
        check = _make_check(
            checkpoint_mode="subset",
            model="llama3-70b",  # any model
            num_processes=8,
            mock_logger=mock_log,
        )
        result = check.closed_mpi_processes()
        assert result is True, "subset mode with correct 8 processes should pass"
        assert mock_log.errors == [], f"Expected no errors, got: {mock_log.errors}"


class TestBug03NonSubsetBranch:
    """Non-subset branch must delegate to Config.get_closed_mpi_processes (D-C4)."""

    def test_bug03_non_subset_branch_delegates_to_config(self):
        """llama3-70b with num_processes=64 must pass (CLOSED_MPI_PROCESSES['70b']==64)."""
        mock_log = MockLogger()
        check = _make_check(
            checkpoint_mode="combined",
            model="llama3-70b",
            num_processes=64,
            mock_logger=mock_log,
        )
        result = check.closed_mpi_processes()
        assert result is True, (
            "70b model with 64 processes should pass. "
            f"Errors: {mock_log.errors}"
        )
        assert mock_log.errors == []

    def test_bug03_non_subset_branch_violates_for_wrong_count(self):
        """llama3-70b with num_processes=32 must return False + violation."""
        mock_log = MockLogger()
        check = _make_check(
            checkpoint_mode="combined",
            model="llama3-70b",
            num_processes=32,
            mock_logger=mock_log,
        )
        result = check.closed_mpi_processes()
        assert result is False, "70b model with 32 processes should fail"
        assert any(
            "[4.6.1 checkpointClosedMpiProcesses]" in e for e in mock_log.errors
        ), f"Expected rule-ID-tagged violation. Got: {mock_log.errors}"
        assert any("70b" in e and "64" in e and "32" in e for e in mock_log.errors), (
            f"Violation message should mention model '70b', required '64', got '32'. "
            f"Got: {mock_log.errors}"
        )
