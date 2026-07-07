#!/usr/bin/env python3
"""
Tests for mlpstorage_py.submission_checker.tools.code_image.capture_or_verify_code_image.

Scope AFTER Plan 06-02 Task 5: only the gating (D-10), env-var validation
(D-04, D-05), and the consensus INLINE `.`/`..` path-traversal guard
(T-02-02-05) remain here. The capture/verify behavioral tests were retired
alongside CAPVER-03 + UX-01 and moved to
`mlpstorage_py/tests/test_capture_or_verify_pool.py` under
`class TestCaptureOrVerifyPool`.

Run with:
    pytest mlpstorage_py/tests/test_capture_or_verify_code_image.py -v
"""

from types import SimpleNamespace

import pytest

from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
    _SUBMITTER_NAME_RE,
    _RESERVED_PATH_SEGMENTS,
)


# ---------------------------------------------------------------------------
# MockLogger that captures status/error calls for assertion.
# ---------------------------------------------------------------------------

class MockLogger:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.warnings = []
        self.infos = []
        self.debugs = []

    def status(self, msg, *args):
        self.statuses.append(msg % args if args else msg)

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)

    # Phase 1 verbose levels (unused here but kept for compatibility)
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def log():
    return MockLogger()


def _make_args(*, mode, command, results_dir, benchmark="training", model="unet3d", orgname=None):
    return SimpleNamespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
        orgname=orgname,   # HARDEN-03: args-first orgname source (main.py LAY-03 hook)
    )


# ---------------------------------------------------------------------------
# Module-level constant sanity
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_submitter_name_regex_compiled(self):
        assert _SUBMITTER_NAME_RE.match("acme_corp.v1-2") is not None
        assert _SUBMITTER_NAME_RE.match("bad name") is None
        assert _SUBMITTER_NAME_RE.match("path/with/slash") is None

    def test_reserved_path_segments(self):
        assert _RESERVED_PATH_SEGMENTS == frozenset({".", ".."})

    def test_regex_accepts_dot_and_dotdot(self):
        # The regex `^[A-Za-z0-9._-]+$` literally matches `.` and `..` —
        # this is exactly why the additional reserved-segments guard is needed.
        assert _SUBMITTER_NAME_RE.match(".") is not None
        assert _SUBMITTER_NAME_RE.match("..") is not None


# ---------------------------------------------------------------------------
# Gating contract (D-10) — no env reads, no fs ops for non-submission modes
# ---------------------------------------------------------------------------

class TestGatingContract:
    def test_whatif_returns_none(self, tmp_path, log):
        args = _make_args(mode="whatif", command="run", results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None
        assert log.statuses == []
        assert log.errors == []

    @pytest.mark.parametrize("mode", [
        "reports", "validate", "history", "lockfile", "version", "rules-coverage",
    ])
    def test_non_submission_modes_return_none(self, tmp_path, log, mode):
        args = _make_args(mode=mode, command="run", results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None

    @pytest.mark.parametrize("command", [
        "configview", "validate", "datasize-something-else",
    ])
    def test_non_submission_commands_return_none(self, tmp_path, log, command):
        # mode is closed but command is not in {datasize, datagen, run} → no-op
        args = _make_args(mode="closed", command=command, results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None


# ---------------------------------------------------------------------------
# Env-var fail-fast (D-04, D-05)
# ---------------------------------------------------------------------------

class TestEnvVarFailFast:
    def test_missing_orgname_raises_configuration_error(self, tmp_path, log):
        args = _make_args(mode="closed", command="datagen", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        assert "MLPSTORAGE_ORGNAME" in str(exc_info.value)
        assert exc_info.value.parameter == "MLPSTORAGE_ORGNAME"
        assert "mlpstorage init" in (exc_info.value.suggestion or "")

    def test_missing_systemname_raises_configuration_error(self, tmp_path, log):
        args = _make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)
        assert exc_info.value.parameter == "MLPSTORAGE_SYSTEMNAME"

    def test_args_orgname_satisfies_without_env_var(self, tmp_path, log):
        """HARDEN-03: args.orgname (populated by main._main_impl's LAY-03 gate)
        satisfies the orgname requirement when MLPSTORAGE_ORGNAME is unset."""
        args = _make_args(mode="closed", command="datagen", results_dir=tmp_path, orgname="acme")
        # No MLPSTORAGE_ORGNAME in env — args.orgname must satisfy the gate.
        result = capture_or_verify_code_image(args, {}, log)
        assert result is not None, "expected Path result; got None (helper unexpectedly gated off)"
        # Downstream consumers depend on _validated_orgname being stashed.
        assert getattr(args, "_validated_orgname", None) == "acme"

    def test_args_systemname_satisfies_without_env_var(self, tmp_path, log):
        """HARDEN-03 (symmetric): args.systemname satisfies the OPEN-mode
        SYSTEMNAME requirement when MLPSTORAGE_SYSTEMNAME is unset."""
        args = _make_args(mode="open", command="datagen", results_dir=tmp_path, orgname="acme")
        args.systemname = "sys1"
        result = capture_or_verify_code_image(args, {}, log)
        assert result is not None
        assert getattr(args, "_validated_systemname", None) == "sys1"

    def test_e101_only_when_both_args_and_env_absent(self, tmp_path, log):
        """HARDEN-03 boundary: E101 fires ONLY when neither args.orgname
        nor MLPSTORAGE_ORGNAME env var is set."""
        args = _make_args(mode="closed", command="datagen", results_dir=tmp_path)
        args.orgname = None  # explicit: both args and env are absent
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        assert "MLPSTORAGE_ORGNAME" in str(exc_info.value)
        assert exc_info.value.parameter == "MLPSTORAGE_ORGNAME"

    def test_orgname_with_space_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "bad name"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "Rules.md" in str(exc_info.value)

    def test_orgname_with_slash_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "evil/path"}
        with pytest.raises(ConfigurationError):
            capture_or_verify_code_image(args, env, log)


# ---------------------------------------------------------------------------
# INLINE path-traversal guard (CONSENSUS FINDING — T-02-02-05)
# ---------------------------------------------------------------------------

class TestPathTraversalGuard:
    def test_orgname_dot_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {"MLPSTORAGE_ORGNAME": "."}, log)
        msg = str(exc_info.value)
        assert "'.' and '..' are reserved path segments" in msg

    def test_orgname_dotdot_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {"MLPSTORAGE_ORGNAME": ".."}, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)

    def test_systemname_dot_rejected(self, tmp_path, log):
        args = _make_args(mode="open", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "."}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)

    def test_systemname_dotdot_rejected(self, tmp_path, log):
        args = _make_args(mode="open", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": ".."}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Capture / verify behavioral tests moved to Plan 06-02:
#
# The pre-Phase-6 legacy `code/` tree-shape assertions (`TestCapturePath`) and
# the hash-mismatch-raise assertions (`TestVerifyPath`) were retired here
# per Plan 06-02 Task 5 alongside CAPVER-03 + UX-01. The replacement coverage
# lives in `mlpstorage_py/tests/test_capture_or_verify_pool.py` under
# `class TestCaptureOrVerifyPool`:
#   - fresh-tree capture: test_fresh_tree_creates_pool_and_pointer
#   - match / reuse:      test_second_call_with_matching_hash_returns_existing_pool_dir_no_new_capture
#   - source change:      test_source_change_creates_second_pool_dir_alongside_first
#   - CAPVER-03 no-raise: test_source_change_does_NOT_raise_CodeImageError
#   - POOL-04 dedup:      test_closed_then_open_same_source_reuses_pool
#   - POOL-03 org isolation: test_two_orgs_maintain_separate_pool_dirs
#   - PTR-01 pointer:     test_pointer_written_after_run_leaf_created
# The retired D-21 "delete `code/` and re-run to re-capture" recovery message
# also no longer applies — Phase 6 refuses legacy `code/` layouts via
# `LegacyLayoutDetected` (D-63); Phase 7 owns the migration.
# ---------------------------------------------------------------------------
