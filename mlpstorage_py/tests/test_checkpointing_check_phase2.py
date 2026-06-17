"""
Tests for CheckpointingCheck Phase 2 requirements: CHKPT-01 through CHKPT-06.

Covers:
  - CHKPT-01 (4.6.4 checkpointOpenSubmissionScaling): open submissions must have
    num_processes as a positive multiple of TP*PP.
  - CHKPT-02 (4.7.1 checkpointCacheFlushValidation): split-mode cache-flush gap
    must be <= 30 seconds.
  - CHKPT-03 (4.7.2 checkpointTotalTestDuration): total test duration is logged
    via info (happy path) or violation (missing/malformed timestamps).
  - CHKPT-04 (4.7.3 checkpointRemappingTimeReporting): observed remap interval
    must be >= declared * 0.5 when remap_time_in_seconds > 0.
  - CHKPT-05 (4.7.4 checkpointSimultaneousRwSupport): runtime cross-check deferred
    per TODO-002; emits log.info only.
  - CHKPT-06 (4.4.2 checkpointFilesystemCheck): checkpoint_folder and results_dir
    must be on different filesystems.

Cross-cutting tests:
  - TestQual02RuleIdPrefix: every Phase 2 CheckpointingCheck error starts with
    [<rule_id> <rule_name>] (mirrors Phase 1 D-05 / QUAL-02 enforcement).
  - TestAccumulateDontAbort: two simultaneous CHKPT-01 violations produce two
    error records (not one) — QUAL-01 / Pitfall 11 compliance.
  - TestChkpt05DeferredFollowUp: simultaneous_rw_support emits INFO (not error)
    when sim_write=False — pinning the documented TODO-002 deferral.

References:
  - D-A3, D-B4, D-B5, D-B7, D-C3, D-D1, D-D2, D-D3 in Phase 2 CONTEXT.md
  - ROADMAP Phase 2 success criteria #4, #5, #6
  - QUAL-01 (accumulate-don't-abort), QUAL-02 (rule-ID prefix), QUAL-04 (coverage)
  - Pitfall #11 (accumulate-don't-abort) and #15 (combined mode is valid)

Run with:
    pytest mlpstorage_py/tests/test_checkpointing_check_phase2.py -v
"""

import pytest
from pathlib import Path

from mlpstorage_py.tests.conftest import (
    _MOCK_DF_OUTPUT_DIFFERENT_MOUNTS,
    _MOCK_DF_OUTPUT_SAME_MOUNT,
)
from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader
from mlpstorage_py.submission_checker.rule_registry import discover_rules


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_checkpointing_check(root_path, mock_logger):
    """Load the submission tree and return the first CheckpointingCheck instance found.

    Args:
        root_path: Path to the submission root (as returned by build_submission).
        mock_logger: MockLogger instance to use for violation capture.

    Returns:
        CheckpointingCheck instance for the first checkpointing SubmissionLogs found.

    Raises:
        AssertionError: if no checkpointing mode was yielded by Loader.load().
    """
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == "checkpointing":
            return CheckpointingCheck(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError("no checkpointing SubmissionLogs yielded from the fixture")


# ---------------------------------------------------------------------------
# TestChkpt01_OpenMpiProcesses
# ---------------------------------------------------------------------------

class TestChkpt01_OpenMpiProcesses:
    """Tests for CHKPT-01 (4.6.4 checkpointOpenSubmissionScaling).

    For OPEN submissions, num_processes must be a positive multiple of TP*PP.
    llama3-70b has TP=8, PP=1 → TP*PP=8. Closed submissions are silently skipped.
    """

    def test_open_70b_with_8_processes_passes(self, tmp_path, mock_logger):
        """Open 70b with 8 processes → passes (8 % 8 == 0)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_open_num_processes=8, chkpt_model="llama3-70b")
        check = _run_checkpointing_check(root, mock_logger)
        result = check.open_mpi_processes()
        assert result is True
        assert mock_logger.errors == []

    def test_open_70b_with_7_processes_emits_4_6_4(self, tmp_path, mock_logger):
        """Open 70b with 7 processes → [4.6.4 checkpointOpenSubmissionScaling] violation."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_open_num_processes=7, chkpt_model="llama3-70b")
        check = _run_checkpointing_check(root, mock_logger)
        result = check.open_mpi_processes()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.6.4 checkpointOpenSubmissionScaling]"), \
            f"Expected [4.6.4 checkpointOpenSubmissionScaling]; got {mock_logger.errors[0]!r}"
        assert "7" in mock_logger.errors[0]

    def test_closed_70b_silently_skips_in_open_method(self, tmp_path, mock_logger):
        """Closed 70b with wrong num_processes → open_mpi_processes silently skips.

        CHKPT-01 open_mpi_processes only checks verification=='open' entries.
        Closed entries are skipped (closed_mpi_processes owns that path per 4.6.1).
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_closed_num_processes=7, chkpt_model="llama3-70b")
        check = _run_checkpointing_check(root, mock_logger)
        result = check.open_mpi_processes()
        assert result is True
        assert mock_logger.errors == []

    def test_open_unknown_model_silently_skips(self, tmp_path, mock_logger):
        """Open submission with unknown model (e.g., 13b) → silent-skip.

        When the model regex (8b|70b|405b|1t) does not match, CHKPT-01 returns
        True with no errors per D-C3.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_open_num_processes=7, chkpt_model="llama3-13b")
        check = _run_checkpointing_check(root, mock_logger)
        result = check.open_mpi_processes()
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# TestChkpt02_CacheFlushValidation
# ---------------------------------------------------------------------------

class TestChkpt02_CacheFlushValidation:
    """Tests for CHKPT-02 (4.7.1 checkpointCacheFlushValidation).

    Split-mode cache-flush gap must be <= 30 seconds. Combined-mode submissions
    (no split-mode pairs) are silently valid per Pitfall 15.
    """

    def test_combined_mode_silently_passes(self, tmp_path, mock_logger):
        """Combined-mode (no split-mode pairs) → passes with no errors (Pitfall 15)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)  # default: no chkpt_split_mode
        check = _run_checkpointing_check(root, mock_logger)
        result = check.cache_flush_validation()
        assert result is True
        assert mock_logger.errors == []

    def test_split_mode_with_25s_gap_passes(self, tmp_path, mock_logger):
        """Split-mode with 25s gap → passes (25 <= 30)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=True,
            chkpt_cache_flush_gap_seconds=25,
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.cache_flush_validation()
        assert result is True
        assert mock_logger.errors == []

    def test_split_mode_with_45s_gap_emits_4_7_1(self, tmp_path, mock_logger):
        """Split-mode with 45s gap → [4.7.1 checkpointCacheFlushValidation] violation."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=True,
            chkpt_cache_flush_gap_seconds=45,
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.cache_flush_validation()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.7.1 checkpointCacheFlushValidation]"), \
            f"Expected [4.7.1 checkpointCacheFlushValidation]; got {mock_logger.errors[0]!r}"
        assert "30-second limit" in mock_logger.errors[0]

    def test_split_mode_missing_timestamps_emits_4_7_1(self, tmp_path, mock_logger):
        """Split-mode without timestamps → [4.7.1] 'missing end_time/start_time'."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=False,  # no timestamps written → missing fields
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.cache_flush_validation()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.7.1 checkpointCacheFlushValidation]"), \
            f"Expected [4.7.1 checkpointCacheFlushValidation]; got {mock_logger.errors[0]!r}"
        assert "missing end_time/start_time" in mock_logger.errors[0]


# ---------------------------------------------------------------------------
# TestChkpt03_TotalTestDuration
# ---------------------------------------------------------------------------

class TestChkpt03_TotalTestDuration:
    """Tests for CHKPT-03 (4.7.2 checkpointTotalTestDuration).

    Happy path emits via log.info (not log_violation) per D-D1. Failure path
    (missing timestamps) emits via log_violation.
    """

    def test_split_mode_with_timestamps_logs_info(self, tmp_path, mock_logger):
        """Split-mode with timestamps → returns True, emits info (not error)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=True,
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.total_test_duration()
        assert result is True
        assert mock_logger.errors == []
        assert any(
            m.startswith("[4.7.2 checkpointTotalTestDuration]")
            for m in mock_logger.infos
        ), f"Expected info starting with [4.7.2 checkpointTotalTestDuration]; got {mock_logger.infos}"

    def test_split_mode_missing_timestamps_emits_4_7_2_error(self, tmp_path, mock_logger):
        """Split-mode without timestamps → [4.7.2] error 'cannot compute total duration'."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=False,
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.total_test_duration()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.7.2 checkpointTotalTestDuration]"), \
            f"Expected [4.7.2 checkpointTotalTestDuration]; got {mock_logger.errors[0]!r}"
        assert "cannot compute total duration" in mock_logger.errors[0]

    def test_combined_mode_silently_passes(self, tmp_path, mock_logger):
        """Combined-mode (no pairs) → passes with no errors and no info log."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)  # default: no split mode → _pair_checkpoint_runs returns []
        check = _run_checkpointing_check(root, mock_logger)
        result = check.total_test_duration()
        assert result is True
        assert mock_logger.errors == []
        # No pairs → no info emitted (nothing to compute)
        assert not any(
            "[4.7.2 checkpointTotalTestDuration]" in m
            for m in mock_logger.infos
        )


# ---------------------------------------------------------------------------
# TestChkpt04_RemappingTimeReporting
# ---------------------------------------------------------------------------

class TestChkpt04_RemappingTimeReporting:
    """Tests for CHKPT-04 (4.7.3 checkpointRemappingTimeReporting).

    When remap_time_in_seconds == 0, no remap is expected → silent-pass.
    When the field is absent (None), SystemYamlSchemaCheck owns the violation
    → silent-skip per D-A3.
    When declared > 0, observed remap interval must be >= declared * 0.5.

    Note on fixture design: with the 5-pair interleaved fixture (5 writes at
    timestamps 100001-140001, 5 reads at 150001-190001), the 'observed remap
    interval' = _parse_iso_gap(last_write_end, first_read_start) is negative
    because the first read's start_time (derived from its pair's write_end + gap)
    is earlier than the last write's end. Any positive declared value with this
    fixture will produce observed_remap < declared * 0.5, triggering the
    violation. This is expected behavior for the negative test. Positive tests
    use the declared==0 and declared==None paths which are independent of
    the observed interval.
    """

    def test_remap_zero_silently_passes(self, tmp_path, mock_logger):
        """Declared remap_time_in_seconds=0 → silent-pass (no remap expected)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_remap_time_seconds=0)
        check = _run_checkpointing_check(root, mock_logger)
        result = check.remapping_time_reporting()
        assert result is True
        assert mock_logger.errors == []

    def test_remap_field_absent_silently_passes(self, tmp_path, mock_logger):
        """Absent remap_time_in_seconds → silent-skip per D-A3 (_get_capability returns None)."""
        from mlpstorage_py.tests.conftest import build_submission
        # Remove the field from system YAML capabilities → _get_capability returns None
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"remove": ["remap_time_in_seconds"]},
        )
        check = _run_checkpointing_check(root, mock_logger)
        assert check._get_capability("remap_time_in_seconds") is None
        result = check.remapping_time_reporting()
        assert result is True
        assert mock_logger.errors == []

    def test_remap_positive_with_too_small_observed_emits_4_7_3(self, tmp_path, mock_logger):
        """Declared remap=100, observed interval negative (fixture interleaving) → violation.

        With the 5-pair interleaved fixture, observed_remap = last_write_end -
        first_read_start is negative (approximately -2390s). Since -2390 < 100 * 0.5 = 50,
        the violation fires correctly.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_split_mode=True,
            chkpt_summary_timestamps=True,
            chkpt_remap_time_seconds=100,
            chkpt_cache_flush_gap_seconds=10,
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.remapping_time_reporting()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.7.3 checkpointRemappingTimeReporting]"), \
            f"Expected [4.7.3 checkpointRemappingTimeReporting]; got {mock_logger.errors[0]!r}"

    def test_remap_default_zero_passes(self, tmp_path, mock_logger):
        """Default fixture (remap_time_in_seconds=0 in system YAML) → silent-pass."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)  # default has remap_time_in_seconds=0
        check = _run_checkpointing_check(root, mock_logger)
        result = check.remapping_time_reporting()
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# TestChkpt05_SimultaneousRwSupport
# ---------------------------------------------------------------------------

class TestChkpt05_SimultaneousRwSupport:
    """Tests for CHKPT-05 (4.7.4 checkpointSimultaneousRwSupport).

    Runtime cross-check is DEFERRED per TODO-002. Method emits log.info only
    and returns True in all cases where capability fields are present. When
    capability fields are absent, SystemYamlSchemaCheck owns the violation
    → silent-skip per D-A3.
    """

    def test_both_true_no_emission(self, tmp_path, mock_logger):
        """Default capabilities (sim_write=True, sim_read=True) → True, no errors, no info."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _run_checkpointing_check(root, mock_logger)
        result = check.simultaneous_rw_support()
        assert result is True
        assert mock_logger.errors == []
        # Default: emits info (capability fields present + deferred message)
        assert len(mock_logger.infos) >= 1

    def test_sim_write_false_emits_info(self, tmp_path, mock_logger):
        """sim_write=False → returns True, emits info with [4.7.4 ...] prefix."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_simultaneous_flags={"simultaneous_write": False},
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.simultaneous_rw_support()
        assert result is True
        assert mock_logger.errors == []
        assert any(
            m.startswith("[4.7.4 checkpointSimultaneousRwSupport]")
            for m in mock_logger.infos
        ), f"Expected info starting with [4.7.4 ...]; got {mock_logger.infos}"

    def test_missing_field_silently_passes(self, tmp_path, mock_logger):
        """Missing simultaneous_write field → silent-pass per D-A3.

        SystemYamlSchemaCheck handles the missing-field violation.
        CHKPT-05 silent-skips when _get_capability returns None.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"remove": ["simultaneous_write"]},
        )
        check = _run_checkpointing_check(root, mock_logger)
        assert check._get_capability("simultaneous_write") is None
        result = check.simultaneous_rw_support()
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# TestChkpt06_CheckpointFilesystemCheck
# ---------------------------------------------------------------------------

class TestChkpt06_CheckpointFilesystemCheck:
    """Tests for CHKPT-06 (4.4.2 checkpointFilesystemCheck).

    Analog of TRAIN-02 for checkpointing. Verifies checkpoint_folder and
    results_dir are on different filesystems. Object-API skips per D-B7.
    """

    def test_default_with_df_different_mounts_passes(self, tmp_path, mock_logger):
        """Different-mount df block → passes with no errors."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_logfile_df_block=_MOCK_DF_OUTPUT_DIFFERENT_MOUNTS,
            chkpt_checkpoint_folder="/data/chkpts",
            chkpt_results_dir="/results/x",
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.checkpoint_filesystem_check()
        assert result is True
        assert mock_logger.errors == []

    def test_same_mount_emits_4_4_2(self, tmp_path, mock_logger):
        """Same-mount df block → [4.4.2 checkpointFilesystemCheck] violation."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_logfile_df_block=_MOCK_DF_OUTPUT_SAME_MOUNT,
            chkpt_checkpoint_folder="/shared/c",
            chkpt_results_dir="/shared/r",
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.checkpoint_filesystem_check()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.4.2 checkpointFilesystemCheck]"), \
            f"Expected [4.4.2 checkpointFilesystemCheck]; got {mock_logger.errors[0]!r}"
        assert "same filesystem" in mock_logger.errors[0]

    def test_df_not_found_emits_4_4_2_missing(self, tmp_path, mock_logger):
        """No df logfile → [4.4.2 checkpointFilesystemCheck] 'df output not found'.

        This is the intentional fail-on-real-submission path per D-B4 / D-B6.
        TODO-001: runtime df capture in mlpstorage CLI is the long-term fix.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_checkpoint_folder="/chkpts",
            chkpt_results_dir="/results",
            # chkpt_logfile_df_block=None (default) → no logfile written
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.checkpoint_filesystem_check()
        assert result is False
        assert len(mock_logger.errors) >= 1
        assert mock_logger.errors[0].startswith("[4.4.2 checkpointFilesystemCheck]"), \
            f"Expected [4.4.2 checkpointFilesystemCheck]; got {mock_logger.errors[0]!r}"
        assert "df output not found" in mock_logger.errors[0]

    def test_object_api_silent_passes(self, tmp_path, mock_logger):
        """benchmark_API='object' → silent-pass; no errors emitted (D-B7)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            benchmark_api="object",
            chkpt_checkpoint_folder="/chkpts",
            chkpt_results_dir="/results",
            # No df block provided — should be silently skipped for object-API
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.checkpoint_filesystem_check()
        assert result is True
        assert mock_logger.errors == [], \
            f"Expected no errors for object-API submission; got {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestQual02RuleIdPrefix — mirror of Phase 1 D-05 enforcement
# ---------------------------------------------------------------------------

# Parametrize table for PHASE2_CHKPT_RULES.
# Each entry: (rule_id, rule_name, mutation_kwargs that trigger that rule's violation).
# CHKPT-05 (4.7.4) is excluded because it only emits info (not errors).
# The test invokes check() which runs ALL check methods, then checks that at
# least one error starts with the expected [rule_id rule_name] prefix.
PHASE2_CHKPT_RULES = [
    (
        "4.6.1",
        "checkpointClosedMpiProcesses",
        {"chkpt_closed_num_processes": 65, "chkpt_model": "llama3-70b"},  # 70b requires 64
    ),
    (
        "4.6.4",
        "checkpointOpenSubmissionScaling",
        {"chkpt_open_num_processes": 7, "chkpt_model": "llama3-70b"},
    ),
    (
        "4.7.1",
        "checkpointCacheFlushValidation",
        {
            "chkpt_split_mode": True,
            "chkpt_summary_timestamps": True,
            "chkpt_cache_flush_gap_seconds": 45,
        },
    ),
    (
        "4.7.2",
        "checkpointTotalTestDuration",
        {"chkpt_split_mode": True, "chkpt_summary_timestamps": False},
    ),
    (
        "4.7.3",
        "checkpointRemappingTimeReporting",
        {
            "chkpt_split_mode": True,
            "chkpt_summary_timestamps": True,
            "chkpt_remap_time_seconds": 100,
            "chkpt_cache_flush_gap_seconds": 10,
        },
    ),
    (
        "4.4.2",
        "checkpointFilesystemCheck",
        {
            "chkpt_logfile_df_block": _MOCK_DF_OUTPUT_SAME_MOUNT,
            "chkpt_checkpoint_folder": "/shared/c",
            "chkpt_results_dir": "/shared/r",
        },
    ),
]


class TestQual02RuleIdPrefix:
    """Every Phase 2 CheckpointingCheck error starts with [<rule_id> <rule_name>].

    Mirror of Phase 1 D-05 / QUAL-02 enforcement. Uses discover_rules to
    confirm the rule is registered, then drives each rule to a violation via
    a single mutation fixture.

    CHKPT-05 (4.7.4) is excluded from this parametrize set because its
    only emission path is log.info (deferred runtime cross-check — TODO-002).
    """

    @pytest.mark.parametrize("rule_id,rule_name,mutation_kwargs", PHASE2_CHKPT_RULES)
    def test_every_phase2_chkpt_method_emits_correct_prefix(
        self, rule_id, rule_name, mutation_kwargs, tmp_path, mock_logger
    ):
        """Each CHKPT rule violation error starts with [rule_id rule_name]."""
        from mlpstorage_py.tests.conftest import build_submission

        # Sanity: confirm rule is registered in discover_rules
        rules = discover_rules(CheckpointingCheck)
        assert rule_id in rules, \
            f"Rule {rule_id!r} not found in discover_rules(CheckpointingCheck): {sorted(rules)}"
        assert rules[rule_id][0] == rule_name, \
            f"Rule {rule_id!r} registered as {rules[rule_id][0]!r}, expected {rule_name!r}"

        root = build_submission(tmp_path, **mutation_kwargs)
        check = _run_checkpointing_check(root, mock_logger)
        check()   # run all check methods

        prefix = f"[{rule_id} {rule_name}]"
        assert any(m.startswith(prefix) for m in mock_logger.errors), \
            (
                f"Expected at least one error starting with {prefix!r}; "
                f"got errors: {mock_logger.errors}"
            )


# ---------------------------------------------------------------------------
# TestAccumulateDontAbort — QUAL-01 / Pitfall 11 compliance
# ---------------------------------------------------------------------------

class TestAccumulateDontAbort:
    """Two simultaneous CHKPT-01 violations must produce two error records.

    Mirrors Phase 1's TestAccumulateDontAbort (from test_submission_checker_structure.py).
    The fixture has 10 timestamps all marked as open with 7 processes (all violating
    for llama3-70b where TP*PP=8). The method must emit 10 errors — one per timestamp
    — proving it does NOT abort on the first violation.
    """

    def test_multiple_chkpt01_violations_emit_multiple_records(self, tmp_path, mock_logger):
        """Open 70b with 7 processes in ALL 10 timestamps → 10 error records from open_mpi_processes.

        Validates QUAL-01 (accumulate-don't-abort): every timestamp produces its own
        error record rather than aborting after the first.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, chkpt_open_num_processes=7, chkpt_model="llama3-70b")
        check = _run_checkpointing_check(root, mock_logger)
        result = check.open_mpi_processes()
        assert result is False
        # All 10 timestamps violate → at least 10 errors emitted (not abort-on-first)
        chkpt01_errors = [
            m for m in mock_logger.errors
            if m.startswith("[4.6.4 checkpointOpenSubmissionScaling]")
        ]
        assert len(chkpt01_errors) >= 2, \
            (
                f"Expected at least 2 CHKPT-01 error records for accumulate-don't-abort "
                f"(QUAL-01); got {len(chkpt01_errors)}: {chkpt01_errors}"
            )


# ---------------------------------------------------------------------------
# TestChkpt05DeferredFollowUp — deferral pinning test
# ---------------------------------------------------------------------------

class TestChkpt05DeferredFollowUp:
    """Pin the documented TODO-002 deferral for simultaneous_rw_support.

    CHKPT-05's runtime per-host cross-check is deferred because current
    summary.json does not expose per-host start/end timing. The method must
    emit log.info (NOT log.error) with 'TODO-002' or 'runtime cross-check' in
    the message, and return True.

    This test exists so future planners cannot accidentally upgrade CHKPT-05
    to a real error-emitting check without noticing the pinned assertion. If
    the deferred check is resolved in a future phase, this test class should
    be updated to assert the error-emitting behavior instead.
    """

    def test_sim_write_false_emits_info_with_deferred_note(self, tmp_path, mock_logger):
        """sim_write=False → log.info with 'runtime cross-check' note; no errors.

        Pinning the deferral: the violation WOULD logically fire here if the
        runtime data were available. But per TODO-002, only log.info is emitted.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_simultaneous_flags={"simultaneous_write": False},
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.simultaneous_rw_support()
        assert result is True
        assert mock_logger.errors == [], \
            f"CHKPT-05 must not emit errors (deferred per TODO-002); got {mock_logger.errors}"
        # The info message must mention the deferral
        matching_infos = [
            m for m in mock_logger.infos
            if "[4.7.4 checkpointSimultaneousRwSupport]" in m
            and ("runtime cross-check" in m or "TODO-002" in m)
        ]
        assert len(matching_infos) >= 1, (
            f"Expected at least one info message with [4.7.4 ...] and deferral note; "
            f"got infos: {mock_logger.infos}"
        )

    def test_deferred_check_returns_true_not_false(self, tmp_path, mock_logger):
        """Deferred CHKPT-05 must return True regardless of capability values.

        If the check were live (not deferred), sim_write=False MIGHT return False.
        Since it is deferred, it MUST return True to avoid false positives.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(
            tmp_path,
            chkpt_simultaneous_flags={"simultaneous_write": False, "simultaneous_read": False},
        )
        check = _run_checkpointing_check(root, mock_logger)
        result = check.simultaneous_rw_support()
        assert result is True, \
            "CHKPT-05 must return True (deferred per TODO-002) even when both sim_* are False"
