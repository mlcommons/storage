"""
Regression tests for the Phase 3 Plan 03-01 DirectoryCheck @rule retrofit.

Covers D-R3 (CONTEXT.md):
  - Every retrofitted method has __rule_id__ and __rule_name__ attributes
    matching the Rules.md §2 binding locked in the plan.
  - The 13 (rule_id, method_name) pairs (2.1.14..2.1.20 + 2.1.22..2.1.27)
    are discoverable via discover_rules(DirectoryCheck).
  - 2.1.27 directoryDiagram is bound via @rule but NOT registered in
    init_checks (no per-submission execution).
  - Three behavior-preservation tests prove the violation-vs-pass logic
    is unchanged AND the rule-ID prefix is now emitted on the violation
    record (one §2.1 representative per group: 2.1.14, 2.1.17, 2.1.23).

Run with:
    pytest mlpstorage_py/tests/test_directory_check_retrofit.py -v
"""

import os
import shutil

import pytest
from unittest.mock import MagicMock

from mlpstorage_py.tests.conftest import build_submission
from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader, SubmissionLogs, LoaderMetadata
from mlpstorage_py.submission_checker.rule_registry import discover_rules


# ---------------------------------------------------------------------------
# Locked (rule_id, method_name) pairs from Plan 03-01 <interfaces>
# ---------------------------------------------------------------------------

RETROFITTED_DIRECTORY_METHODS = [
    ("2.1.14", "datagen_files_check"),
    ("2.1.15", "datagen_dlio_config_check"),
    ("2.1.16", "run_results_json_check"),
    ("2.1.17", "run_files_timestamp_check"),
    ("2.1.18", "run_duration_valid_check"),
    ("2.1.19", "run_files_check"),
    ("2.1.20", "run_dlio_config_check"),
    ("2.1.22", "checkpointing_results_json_check"),
    ("2.1.23", "checkpointing_timestamps_check"),
    ("2.1.24", "checkpointing_timestamp_gap_check"),
    ("2.1.25", "checkpointing_files_check"),
    ("2.1.26", "checkpointing_dlio_config_check"),
    ("2.1.27", "directory_diagram_check"),
]

# Rule names for each rule ID (locked by Plan 03-01 <interfaces> table)
_RULE_NAMES = {
    "2.1.14": "datagenFiles",
    "2.1.15": "datagenDlioConfig",
    "2.1.16": "runResultsJson",
    "2.1.17": "runTimestamps",
    "2.1.18": "runTimestampGap",
    "2.1.19": "runFiles",
    "2.1.20": "runDlioConfig",
    "2.1.22": "checkpointingResultsJson",
    "2.1.23": "checkpointingTimestamps",
    "2.1.24": "checkpointingTimestampGap",
    "2.1.25": "checkpointingFiles",
    "2.1.26": "checkpointingDlioConfig",
    "2.1.27": "directoryDiagram",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instantiate_directory_check(root_path, mode, mock_logger):
    """Load the submission tree at root_path and return the first
    DirectoryCheck instance for the requested mode ("training" or
    "checkpointing").

    Args:
        root_path: Path to the submission root (as returned by build_submission).
        mode: "training" or "checkpointing".
        mock_logger: MockLogger instance for violation capture.

    Returns:
        DirectoryCheck instance bound to the first matching SubmissionLogs.

    Raises:
        AssertionError: if no SubmissionLogs for the requested mode was yielded.
    """
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == mode:
            return DirectoryCheck(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError(f"no {mode} SubmissionLogs yielded from the fixture at {root_path}")


# ---------------------------------------------------------------------------
# TestRetrofittedMethodsCarryRuleIdAttribute  (D-R3.a — parametrized)
# ---------------------------------------------------------------------------

class TestRetrofittedMethodsCarryRuleIdAttribute:
    """Every retrofitted method must carry __rule_id__ + __rule_name__
    attributes attached by the @rule decorator (Plan 03-01 Tasks 1 + 2).

    Parametrized over the 13 (rule_id, method_name) pairs from
    RETROFITTED_DIRECTORY_METHODS.
    """

    @pytest.mark.parametrize("rule_id,method_name", RETROFITTED_DIRECTORY_METHODS)
    def test_every_retrofitted_method_has_rule_id_attribute(self, rule_id, method_name):
        """The method's __rule_id__ matches the expected ID and __rule_name__ is set."""
        method = getattr(DirectoryCheck, method_name, None)
        assert method is not None, (
            f"DirectoryCheck has no attribute named {method_name!r}; the plan's "
            f"locked (rule_id, method_name) binding is broken."
        )
        assert getattr(method, "__rule_id__", None) == rule_id, (
            f"{method_name}.__rule_id__ is {getattr(method, '__rule_id__', None)!r}; "
            f"expected {rule_id!r}."
        )
        expected_name = _RULE_NAMES[rule_id]
        assert getattr(method, "__rule_name__", None) == expected_name, (
            f"{method_name}.__rule_name__ is {getattr(method, '__rule_name__', None)!r}; "
            f"expected {expected_name!r}."
        )

    def test_discover_rules_returns_all_thirteen(self):
        """discover_rules(DirectoryCheck) returns the full 13-entry mapping."""
        rules = discover_rules(DirectoryCheck)
        want = {rid for rid, _ in RETROFITTED_DIRECTORY_METHODS}
        got = set(rules)
        missing = want - got
        extra = got - want
        assert not missing, f"missing rule IDs from discover_rules: {sorted(missing)}"
        assert not extra, (
            f"unexpected rule IDs in discover_rules (not in plan's locked binding): "
            f"{sorted(extra)}"
        )
        # Per-entry (rule_name, method_name) shape
        for rule_id, method_name in RETROFITTED_DIRECTORY_METHODS:
            rule_name, mname = rules[rule_id]
            assert rule_name == _RULE_NAMES[rule_id]
            assert mname == method_name


# ---------------------------------------------------------------------------
# TestDirectoryDiagramNotInInitChecks  (D-A1 — no per-submission execution)
# ---------------------------------------------------------------------------

class TestDirectoryDiagramNotInInitChecks:
    """2.1.27 directoryDiagram is bound via @rule but MUST NOT be registered
    in init_checks — Rules.md 2.1.27 is a pictorial illustration, not a
    runtime check (CONTEXT.md D-A1).
    """

    def test_training_mode_check_list_excludes_directory_diagram(self, tmp_path, mock_logger):
        """DirectoryCheck instance in training mode does NOT register
        directory_diagram_check in self.checks.
        """
        root = build_submission(tmp_path)
        check = _instantiate_directory_check(root, "training", mock_logger)
        method_names = [c.__name__ for c in check.checks]
        assert "directory_diagram_check" not in method_names, (
            f"directory_diagram_check must NOT be in init_checks; got: {method_names}"
        )

    def test_checkpointing_mode_check_list_excludes_directory_diagram(self, tmp_path, mock_logger):
        """DirectoryCheck instance in checkpointing mode does NOT register
        directory_diagram_check in self.checks.
        """
        root = build_submission(tmp_path)
        check = _instantiate_directory_check(root, "checkpointing", mock_logger)
        method_names = [c.__name__ for c in check.checks]
        assert "directory_diagram_check" not in method_names, (
            f"directory_diagram_check must NOT be in init_checks; got: {method_names}"
        )

    def test_directory_diagram_returns_true_with_no_logging(self, mock_logger):
        """Calling directory_diagram_check on a bare instance returns True
        and emits no log records — proves it's a true no-op.
        """
        check = DirectoryCheck.__new__(DirectoryCheck)
        check.log = mock_logger
        result = check.directory_diagram_check()
        assert result is True
        assert mock_logger.errors == []
        assert mock_logger.warnings == []


# ---------------------------------------------------------------------------
# TestBehaviorPreservation_2_1_14_DatagenFiles
# ---------------------------------------------------------------------------

class TestBehaviorPreservation_2_1_14_DatagenFiles:
    """2.1.14 datagenFiles: missing required files in the datagen timestamp
    directory must emit a [2.1.14 datagenFiles] violation and return False.
    """

    def test_datagen_missing_summary_emits_prefixed_violation(self, tmp_path, mock_logger):
        """Default fixture's datagen timestamp dir lacks *summary.json (only
        metadata.json is written) — datagen_files_check must report it as a
        2.1.14 violation with the locked prefix and return False.
        """
        root = build_submission(tmp_path)
        check = _instantiate_directory_check(root, "training", mock_logger)
        result = check.datagen_files_check()
        assert result is False, (
            "default datagen tree has only metadata.json — datagen_files_check "
            "must fail on missing required files"
        )
        assert any(
            m.startswith("[2.1.14 datagenFiles] ") for m in mock_logger.errors
        ), (
            f"expected at least one error starting with '[2.1.14 datagenFiles] '; "
            f"got {mock_logger.errors!r}"
        )


# ---------------------------------------------------------------------------
# TestBehaviorPreservation_2_1_17_RunTimestamps
# ---------------------------------------------------------------------------

class TestBehaviorPreservation_2_1_17_RunTimestamps:
    """2.1.17 runTimestamps: wrong count (!= 6) must emit a
    [2.1.17 runTimestamps] violation and return False.
    """

    def test_run_timestamps_wrong_count_emits_prefixed_violation(self, mock_logger):
        """5 valid timestamps — count check fires and the error message
        carries the [2.1.17 runTimestamps] prefix.

        Uses the __new__ + manual-attr construction pattern from
        test_directory_check_run_timestamps.py so we don't depend on the
        full loader tree (the format-vs-count branches are the units under
        test, not the loader).
        """
        check = DirectoryCheck.__new__(DirectoryCheck)
        check.log = mock_logger
        check.run_path = "/test/run"
        submissions_logs = MagicMock()
        # 5 valid timestamps — fails the count gate (RUN_TIMESTAMP_COUNT=6)
        submissions_logs.run_files = [
            (None, None, f"20260101_12000{i}") for i in range(5)
        ]
        check.submissions_logs = submissions_logs

        result = check.run_files_timestamp_check()
        assert result is False, "5 timestamps must fail run_files_timestamp_check"
        assert any(
            m.startswith("[2.1.17 runTimestamps] ") for m in mock_logger.errors
        ), (
            f"expected at least one error starting with '[2.1.17 runTimestamps] '; "
            f"got {mock_logger.errors!r}"
        )


# ---------------------------------------------------------------------------
# TestBehaviorPreservation_2_1_23_CheckpointingTimestamps
# ---------------------------------------------------------------------------

class TestBehaviorPreservation_2_1_23_CheckpointingTimestamps:
    """2.1.23 checkpointingTimestamps: wrong count (not 1 or 2) must emit a
    [2.1.23 checkpointingTimestamps] violation and return False.

    Per Rules.md 4.7.1, a CLOSED checkpointing submission has 1 invocation
    (combined write+read) or 2 invocations (write phase + read phase), so
    the directory shape is 1 or 2 timestamp dirs — not 10. See the
    docstring on DirectoryCheck.checkpointing_timestamps_check.
    """

    def test_checkpointing_timestamps_wrong_count_emits_prefixed_violation(
        self, tmp_path, mock_logger
    ):
        """7 timestamp directories in the workload (count not in {1, 2}) —
        the violation carries the [2.1.23 checkpointingTimestamps] prefix.

        ``loader.py:103`` yields ``loader_metadata.folder =
        .../checkpointing/<workload>``, so ``self.checkpointing_path`` IS
        the workload directory. Build a workload tree with 7 timestamp
        subdirs and point ``checkpointing_path`` at it directly.
        """
        workload = tmp_path / "llama3-8b"
        workload.mkdir()
        for i in range(7):
            (workload / f"20260101_12000{i}").mkdir()

        check = DirectoryCheck.__new__(DirectoryCheck)
        check.log = mock_logger
        check.checkpointing_path = str(workload)
        submissions_logs = MagicMock()
        submissions_logs.checkpoint_files = [("x", "y", "z")]
        check.submissions_logs = submissions_logs

        result = check.checkpointing_timestamps_check()
        assert result is False, (
            "7 timestamp dirs must fail checkpointing_timestamps_check "
            "(expected 1 or 2)"
        )
        assert any(
            m.startswith("[2.1.23 checkpointingTimestamps] ") for m in mock_logger.errors
        ), (
            f"expected at least one error starting with "
            f"'[2.1.23 checkpointingTimestamps] '; got {mock_logger.errors!r}"
        )
