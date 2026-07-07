#!/usr/bin/env python3
"""CAP/VALR gating + env-var validation tests for the CLI dispatch helper.

Scope AFTER Plan 06-02 Task 5:
    CAP-07, CAP-08 gating (whatif/reports/validate/etc. + non-submission
        commands under closed|open)
    D-04, D-05 MLPSTORAGE_ORGNAME / MLPSTORAGE_SYSTEMNAME validation
    Path-traversal '.' / '..' rejection (REVIEWS.md consensus finding,
        Gemini + plan-checker — _RESERVED_PATH_SEGMENTS guard).

The capture/verify behavioral tests (CAP-01/02/06 legacy paths and the
VALR-01..04 reject-string assertions) were retired here alongside
CAPVER-03 + UX-01. Replacement coverage lives in
`mlpstorage_py/tests/test_capture_or_verify_pool.py`.

Tests exercise ``capture_or_verify_code_image(args, env, log)`` via direct
in-process invocation with ``tmp_path`` + MockLogger fixtures (CD-02 —
chosen lightweight style, no subprocess / no MPI).

Run with:
    pytest mlpstorage_py/tests/test_cli_code_image.py -v
"""

from types import SimpleNamespace

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
)
from mlpstorage_py.errors import ConfigurationError


# ---------------------------------------------------------------------------
# MockLogger — captures status/warning/error/info/debug calls.
# Mirrors the PATTERNS.md "Imports + MockLogger pattern" with the extra
# ``status`` channel that the Phase 2 helper uses for CAP-06 / VALR-01/03
# success messages.
# ---------------------------------------------------------------------------

class MockLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []
        self.debugs = []
        self.statuses = []

    def debug(self, msg, *a):   self.debugs.append(msg % a if a else msg)
    def info(self, msg, *a):    self.infos.append(msg % a if a else msg)
    def status(self, msg, *a):  self.statuses.append(msg % a if a else msg)
    def warning(self, msg, *a): self.warnings.append(msg % a if a else msg)
    def error(self, msg, *a):   self.errors.append(msg % a if a else msg)
    def verbose(self, *a, **k): pass
    def verboser(self, *a, **k): pass
    def ridiculous(self, *a, **k): pass


@pytest.fixture
def mock_logger():
    return MockLogger()


# ---------------------------------------------------------------------------
# fake_source_root — isolated tmp source tree to keep the live-source hash
# deterministic across capture (shutil.copytree+ignore) and verify
# (compute_code_tree_md5 direct walk). Documented in deferred-items.md as a
# Phase 1 follow-up; the workaround is the same pattern Plan 02-02's tests use.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_source_root(tmp_path, monkeypatch):
    src = tmp_path / "src_root"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname = 'x'\nversion='0.0.1'\n")
    (src / "mlpstorage_py").mkdir()
    (src / "mlpstorage_py" / "__init__.py").write_text("__version__ = '0.0.1'\n")
    (src / "mlpstorage_py" / "stub.py").write_text("X = 1\n")
    monkeypatch.setattr(
        "mlpstorage_py.submission_checker.tools.code_image.find_source_root",
        lambda: src,
    )
    return src


# ---------------------------------------------------------------------------
# make_args helper — small factory matching the helper's args shape.
# ---------------------------------------------------------------------------

def make_args(*, mode, command, results_dir, benchmark="training", model="unet3d"):
    return SimpleNamespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
    )


# ---------------------------------------------------------------------------
# Capture / verify behavioral tests moved to Plan 06-02:
#
# The pre-Phase-6 legacy `code/` tree-shape assertions
# (TestClosedFirstCapture / TestOpenFirstCapture), the "code unchanged" reuse
# assertions (TestRuntimeMatchPasses), and the hash-mismatch reject-string
# assertions (TestRuntimeMismatchCLOSED / TestRuntimeMismatchOPEN) were retired
# here per Plan 06-02 Task 5 alongside CAPVER-03 + UX-01. Replacement coverage
# lives in `mlpstorage_py/tests/test_capture_or_verify_pool.py::TestCaptureOrVerifyPool`:
#   - fresh capture:      test_fresh_tree_creates_pool_and_pointer
#   - closed reuse:       test_second_call_with_matching_hash_returns_existing_pool_dir_no_new_capture
#   - open reuse:         test_open_then_closed_same_source_reuses_pool
#   - source change:      test_source_change_creates_second_pool_dir_alongside_first
#   - CAPVER-03 no-raise: test_source_change_does_NOT_raise_CodeImageError
#   - UX-01 negative:     test_source_change_stderr_does_NOT_contain_retired_reject_string
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestNoTouchSubcommands (CAP-07, CAP-08, TEST-09)
# ---------------------------------------------------------------------------

# Parametrized over the seven non-result-generating modes. The helper must
# return None and perform NO filesystem operations or env reads for each.
_NO_TOUCH_MODES = [
    "whatif",
    "reports",
    "validate",
    "history",
    "lockfile",
    "version",
    "rules-coverage",
]


class TestNoTouchSubcommands:
    """CAP-07/08: helper is a no-op for whatif/validate/reportgen/etc. AND
    for {closed|open} commands that are not in {datasize, datagen, run}.
    """

    @pytest.mark.parametrize("mode", _NO_TOUCH_MODES)
    def test_no_touch(self, tmp_path, mock_logger, mode):
        # An empty env confirms the helper does NOT read MLPSTORAGE_* env vars
        # in the gated-off path (CAP-07/08).
        args = make_args(mode=mode, command="run", results_dir=tmp_path)
        env = {}
        result = capture_or_verify_code_image(args, env, mock_logger)
        assert result is None
        # No subdirectories created under tmp_path.
        assert not (tmp_path / "closed").exists()
        assert not (tmp_path / "open").exists()
        # No logger calls (gate runs before any logging in the helper).
        assert mock_logger.statuses == []
        assert mock_logger.errors == []
        assert mock_logger.warnings == []
        assert mock_logger.infos == []

    def test_no_touch_invalid_command_under_valid_mode(self, tmp_path, mock_logger):
        # Under closed|open mode, command not in {datasize, datagen, run} →
        # helper still returns None and performs no fs/env work.
        args = make_args(mode="closed", command="configview", results_dir=tmp_path)
        env = {}
        result = capture_or_verify_code_image(args, env, mock_logger)
        assert result is None
        assert not (tmp_path / "closed").exists()
        assert mock_logger.errors == []
        assert mock_logger.warnings == []

    def test_no_touch_open_with_configview_command(self, tmp_path, mock_logger):
        # Under open mode, command not in {datasize, datagen, run} →
        # helper still returns None and performs no fs/env work.
        args = make_args(mode="open", command="history", results_dir=tmp_path)
        env = {}
        result = capture_or_verify_code_image(args, env, mock_logger)
        assert result is None
        assert not (tmp_path / "open").exists()
        assert mock_logger.errors == []
        assert mock_logger.warnings == []

    @pytest.mark.parametrize("command", ["datasize", "datagen", "run"])
    def test_gating_passes_for_each_submission_command(
        self, tmp_path, fake_source_root, mock_logger, command
    ):
        # Sanity: each of the three result-generating commands triggers
        # capture-or-verify (returns a Path, creates code/), confirming the
        # gating set membership and that no command in the spec is missed.
        args = make_args(mode="closed", command=command, results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        result = capture_or_verify_code_image(args, env, mock_logger)
        assert result is not None
        assert result.is_dir()


# ---------------------------------------------------------------------------
# TestEnvVarValidation (D-04, D-05)
# ---------------------------------------------------------------------------

class TestEnvVarValidation:
    """Fail-fast on missing or POSIX-invalid MLPSTORAGE_* env vars."""

    def test_missing_orgname_closed(self, tmp_path, mock_logger):
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        msg = str(exc_info.value)
        assert "MLPSTORAGE_ORGNAME" in msg
        # ConfigurationError.suggestion should mention the future setup command.
        suggestion = getattr(exc_info.value, "suggestion", "") or getattr(
            exc_info.value.error, "suggestion", ""
        )
        assert "mlpstorage init" in suggestion, suggestion

    def test_missing_systemname_open(self, tmp_path, mock_logger):
        args = make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)

    def test_invalid_posix_orgname(self, tmp_path, mock_logger):
        # Space is not in [A-Za-z0-9._-].
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "bad name"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        assert "Rules.md §2.1.1" in str(exc_info.value)
        assert "MLPSTORAGE_ORGNAME" in str(exc_info.value)

    def test_invalid_posix_systemname(self, tmp_path, mock_logger):
        # Slash is not in [A-Za-z0-9._-] (path-traversal-adjacent).
        args = make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "with/slash"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        assert "Rules.md §2.1.1" in str(exc_info.value)
        assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestEnvVarPathTraversal — CONSENSUS FINDING (Gemini + plan-checker)
# ---------------------------------------------------------------------------

class TestEnvVarPathTraversal:
    """REVIEWS.md consensus finding: the regex ^[A-Za-z0-9._-]+$ accepts '.' and
    '..' literally. Plan 02 added an inline ``_RESERVED_PATH_SEGMENTS`` guard
    AFTER the regex check. These tests pin that guard for BOTH env vars.

    Substring contract: the helper raises ConfigurationError with a message
    containing the literal substring "'.' and '..' are reserved path segments".
    """

    @pytest.mark.parametrize("bad_value", [".", ".."])
    def test_orgname_dot_raises_configuration_error(
        self, tmp_path, bad_value, mock_logger
    ):
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": bad_value}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        msg = str(exc_info.value)
        assert "'.' and '..' are reserved path segments" in msg
        assert "MLPSTORAGE_ORGNAME" in msg

    @pytest.mark.parametrize("bad_value", [".", ".."])
    def test_systemname_dot_raises_configuration_error(
        self, tmp_path, bad_value, mock_logger
    ):
        args = make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": bad_value}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        msg = str(exc_info.value)
        assert "'.' and '..' are reserved path segments" in msg
        assert "MLPSTORAGE_SYSTEMNAME" in msg

    def test_valid_names_pass_sanity_check(
        self, tmp_path, fake_source_root, mock_logger
    ):
        """Sanity: valid POSIX names that are NOT '.'/'..' must NOT raise.

        Confirms that the rejection in the prior two tests is specifically
        due to the '.'/'..' guard, not a different validation bug.
        """
        args = make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "valid_name", "MLPSTORAGE_SYSTEMNAME": "valid_name"}
        result = capture_or_verify_code_image(args, env, mock_logger)
        assert result is not None
        assert result.exists()

    def test_filesystem_unchanged_after_path_traversal_reject(self, tmp_path, mock_logger):
        """The helper rejects BEFORE any mkdir — filesystem is untouched."""
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "."}
        with pytest.raises(ConfigurationError):
            capture_or_verify_code_image(args, env, mock_logger)
        assert not (tmp_path / "closed").exists()
        assert not (tmp_path / "open").exists()


# ---------------------------------------------------------------------------
# TestBadImageRecovery (D-21) — retired in Plan 06-02 Task 5.
#
# The D-21 "delete `code/` and re-run to re-capture" recovery message applied
# to the legacy single-`code/` layout. Phase 6 replaces that with a
# content-addressed pool at `<results_dir>/<orgname>/code-<hash8>/` (D-64);
# any legacy `code/` present at capture time is refused via
# `LegacyLayoutDetected` (D-63), with Phase 7 owning the migration. The
# missing/malformed .code-hash.json recovery workflow is now covered at
# the pool-scan layer: `_find_matching_pool_image` catches
# MissingHashFile / MalformedHashFile at DEBUG level and continues (skips
# non-conformant pool candidates), so a corrupt pool image never trips the
# runtime capture path — Phase 8's CHECK-02 owns the audit-time
# self-consistency verification.
# ---------------------------------------------------------------------------
