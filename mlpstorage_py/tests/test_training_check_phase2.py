"""
Tests for TrainingCheck Phase 2 requirements: TRAIN-01 + TRAIN-02.

Covers:
  - TRAIN-01 (3.3.4 trainingSingleHostClientLimit): single-host with multiple
    hosts list emits violation; single host with one host passes.
  - TRAIN-02 (3.4.2 trainingMlpstorageFilesystemCheck): different mounts pass;
    same mount emits violation; df-not-found emits violation; object-API silent-pass;
    missing data_dir silent-skip.

References:
  - D-B3, D-B4, D-B6, D-B7 in Phase 2 CONTEXT.md (df parsing, object-API gating)
  - ROADMAP Phase 2 success criteria #2 (TRAIN-01) and #3 (TRAIN-02)
  - QUAL-04 per-requirement positive + negative test coverage

Run with:
    pytest mlpstorage_py/tests/test_training_check_phase2.py -v
"""

import os
import pytest
from pathlib import Path

from mlpstorage_py.tests.conftest import (
    _MOCK_DF_OUTPUT_DIFFERENT_MOUNTS,
    _MOCK_DF_OUTPUT_SAME_MOUNT,
)
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_training_check(root_path, mock_logger):
    """Load the submission tree and return the first TrainingCheck instance found.

    Args:
        root_path: Path to the submission root (as returned by build_submission).
        mock_logger: MockLogger instance to use for violation capture.

    Returns:
        TrainingCheck instance for the first training SubmissionLogs found.

    Raises:
        AssertionError: if no training mode was yielded by Loader.load().
    """
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == "training":
            return TrainingCheck(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError("no training SubmissionLogs yielded from the fixture")


# ---------------------------------------------------------------------------
# TestTrain01_SingleHostClientLimit
# ---------------------------------------------------------------------------

class TestTrain01_SingleHostClientLimit:
    """Tests for TRAIN-01 (3.3.4 trainingSingleHostClientLimit).

    TRAIN-01 fires when summary.json reports num_hosts==1 but the run metadata
    args.hosts list contains more than one entry. Rule ID: 3.3.4.
    """

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        """Default fixture (num_hosts=2 in summary) → TRAIN-01 always skips (no single-host).

        Default build_submission has summary.json num_hosts=2 and metadata hosts=["host1"].
        Since num_hosts != 1, the single-host check never fires.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _run_training_check(root, mock_logger)
        result = check.single_host_client_limit()
        assert result is True
        assert mock_logger.errors == []

    def test_single_host_with_multiple_hosts_emits_3_3_4(self, tmp_path, mock_logger):
        """Single-host run (num_hosts missing from summary → defaults to 1) with 2-host list
        in metadata emits [3.3.4 trainingSingleHostClientLimit] violation.

        Uses missing_summary_field='num_hosts' so summary.get('num_hosts', 1) returns 1,
        simulating a single-host submission. The metadata args.hosts=['host1','host2']
        triggers the violation.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_metadata_hosts=["host1", "host2"],
            missing_summary_field="num_hosts",  # summary defaults to num_hosts=1
        )
        check = _run_training_check(root, mock_logger)
        result = check.single_host_client_limit()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert any(
            m.startswith("[3.3.4 trainingSingleHostClientLimit]")
            for m in mock_logger.errors
        ), f"Expected error starting with [3.3.4 trainingSingleHostClientLimit]; got {mock_logger.errors}"

    def test_multi_host_with_multiple_hosts_passes(self, tmp_path, mock_logger):
        """Multi-host run (num_hosts=2 in summary) with 2 hosts in metadata → passes.

        When summary.num_hosts > 1, TRAIN-01 skip applies — the client count
        is only checked for single-host runs.
        """
        from mlpstorage_py.tests.conftest import build_submission
        # Default fixture: num_hosts=2 in summary, hosts=["host1"] in metadata.
        # Use 2 hosts in metadata to confirm no false positive.
        root = build_submission(tmp_path, run_metadata_hosts=["host1", "host2"])
        check = _run_training_check(root, mock_logger)
        result = check.single_host_client_limit()
        assert result is True
        assert mock_logger.errors == []

    def test_single_host_with_one_host_passes(self, tmp_path, mock_logger):
        """Single-host run (num_hosts missing → 1) with exactly one host → passes.

        The violation is: single-host + MULTIPLE hosts in metadata. One host
        is perfectly fine.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_metadata_hosts=["host1"],
            missing_summary_field="num_hosts",
        )
        check = _run_training_check(root, mock_logger)
        result = check.single_host_client_limit()
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# TestTrain02_MlpstorageFilesystemCheck
# ---------------------------------------------------------------------------

class TestTrain02_MlpstorageFilesystemCheck:
    """Tests for TRAIN-02 (3.4.2 trainingMlpstorageFilesystemCheck).

    TRAIN-02 parses the 'df' block from training_run.stdout.log and verifies
    that data_dir and results_dir are on different filesystems. Object-API
    submissions skip the check entirely per D-B7.
    """

    def test_default_fixture_with_df_block_passes(self, tmp_path, mock_logger):
        """Different-mount df block + different path prefixes → passes (no violation)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_logfile_df_block=_MOCK_DF_OUTPUT_DIFFERENT_MOUNTS,
            run_data_dir="/data/foo",
            run_results_dir="/results/bar",
        )
        check = _run_training_check(root, mock_logger)
        result = check.mlpstorage_filesystem_check()
        assert result is True
        assert mock_logger.errors == []

    def test_same_mount_emits_3_4_2(self, tmp_path, mock_logger):
        """Same-mount df block + shared path prefix → [3.4.2 trainingMlpstorageFilesystemCheck]."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_logfile_df_block=_MOCK_DF_OUTPUT_SAME_MOUNT,
            run_data_dir="/shared/foo",
            run_results_dir="/shared/bar",
        )
        check = _run_training_check(root, mock_logger)
        result = check.mlpstorage_filesystem_check()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[3.4.2 trainingMlpstorageFilesystemCheck]"), \
            f"Expected prefix [3.4.2 trainingMlpstorageFilesystemCheck]; got {mock_logger.errors[0]!r}"
        assert "same filesystem" in mock_logger.errors[0], \
            f"Expected 'same filesystem' in error; got {mock_logger.errors[0]!r}"

    def test_df_not_found_emits_3_4_2_missing(self, tmp_path, mock_logger):
        """No df logfile → [3.4.2 trainingMlpstorageFilesystemCheck] 'df output not found'.

        This is the intentional fail-on-real-submission path per D-B4 / D-B6.
        TODO-001: runtime df capture in mlpstorage CLI is the long-term fix.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_data_dir="/data",
            run_results_dir="/results",
            # run_logfile_df_block=None (default) → no logfile written
        )
        check = _run_training_check(root, mock_logger)
        result = check.mlpstorage_filesystem_check()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[3.4.2 trainingMlpstorageFilesystemCheck]"), \
            f"Expected prefix [3.4.2 trainingMlpstorageFilesystemCheck]; got {mock_logger.errors[0]!r}"
        assert "df output not found" in mock_logger.errors[0], \
            f"Expected 'df output not found' in error; got {mock_logger.errors[0]!r}"

    def test_object_api_silent_passes(self, tmp_path, mock_logger):
        """benchmark_API='object' → silent-pass; no errors emitted (D-B7)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            benchmark_api="object",
            # No df block provided — but should be silently skipped for object-API
        )
        check = _run_training_check(root, mock_logger)
        result = check.mlpstorage_filesystem_check()
        assert result is True
        assert mock_logger.errors == [], \
            f"Expected no errors for object-API submission; got {mock_logger.errors}"

    def test_silent_skip_on_missing_data_dir(self, tmp_path, mock_logger):
        """Missing data_dir in metadata → silent-skip per D-B3 (no error, no warning).

        The _check_filesystem_separation helper returns (True, True) when either
        path is absent. TRAIN-02 should NOT emit a violation — that's the role of
        the companion mlpstorage_path_args check.
        """
        from mlpstorage_py.tests.conftest import build_submission
        # Provide a df block but set data_dir to empty string in metadata.
        # The factory's run_data_dir="" sets args.data_dir="" which the helper
        # treats as falsy → D-B3 silent-skip.
        root = build_submission(
            tmp_path,
            run_logfile_df_block=_MOCK_DF_OUTPUT_DIFFERENT_MOUNTS,
            run_data_dir="",
            run_results_dir="/results",
        )
        check = _run_training_check(root, mock_logger)
        result = check.mlpstorage_filesystem_check()
        # D-B3: empty data_dir → silent-skip → returns True, no error
        assert result is True
        assert mock_logger.errors == [], \
            f"Expected no errors for missing data_dir (D-B3 silent-skip); got {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestTrain02_DfBlockNotFoundIsRuleIdTagged  (ROADMAP success criterion #3)
# ---------------------------------------------------------------------------

class TestTrain02_DfBlockNotFoundIsRuleIdTagged:
    """Verify the 'df output not found' violation message includes the logfile path.

    ROADMAP Phase 2 success criterion #3: submitters can identify the specific
    logfile that failed the check from the violation message.
    """

    def test_df_not_found_violation_includes_logfile_path(self, tmp_path, mock_logger):
        """df-not-found error must include 'training_run.stdout.log' in the message.

        This ensures submitters can grep for the failing logfile path from the
        violation output.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            run_data_dir="/data",
            run_results_dir="/results",
        )
        check = _run_training_check(root, mock_logger)
        check.mlpstorage_filesystem_check()
        assert len(mock_logger.errors) >= 1
        # The violation message must reference the logfile path
        assert any(
            "training_run.stdout.log" in m
            for m in mock_logger.errors
        ), (
            f"Expected 'training_run.stdout.log' in at least one error message; "
            f"got {mock_logger.errors}"
        )
