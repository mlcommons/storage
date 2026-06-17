#!/usr/bin/env python3
"""Phase 2 Plan 02-05 — CAP/VALR contract tests for the CLI dispatch helper.

Covers requirements:
    CAP-01, CAP-02, CAP-06, CAP-07, CAP-08
    VALR-01, VALR-02, VALR-03, VALR-04
    D-04, D-05, D-21
    Path-traversal '.' / '..' rejection (REVIEWS.md consensus finding,
        Gemini + plan-checker — _RESERVED_PATH_SEGMENTS guard).

Tests exercise ``capture_or_verify_code_image(args, env, log)`` via direct
in-process invocation with ``tmp_path`` + MockLogger fixtures (CD-02 —
chosen lightweight style, no subprocess / no MPI).

Run with:
    pytest mlpstorage_py/tests/test_cli_code_image.py -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
    capture_code_image,
    CodeImageError,
    MissingHashFile,
    MalformedHashFile,
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
# TestClosedFirstCapture (CAP-01, CAP-06, TEST-02)
# ---------------------------------------------------------------------------

class TestClosedFirstCapture:
    """CAP-01: first call on closed|datagen captures the image at
    {results_dir}/closed/<orgname>/code/.
    """

    def test_closed_first_capture_creates_code_dir(
        self, tmp_path, fake_source_root, mock_logger
    ):
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        result = capture_or_verify_code_image(args, env, mock_logger)
        expected = tmp_path / "closed" / "acme" / "code"
        assert result == expected
        assert expected.is_dir()
        assert (expected / ".code-hash.json").is_file()

    def test_closed_first_capture_logs_absolute_path(
        self, tmp_path, fake_source_root, mock_logger
    ):
        # CAP-06: log starts with "Captured code image at " followed by the
        # absolute code/ path.
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        capture_or_verify_code_image(args, env, mock_logger)
        expected = tmp_path / "closed" / "acme" / "code"
        assert any(
            s.startswith("Captured code image at ") and str(expected) in s
            for s in mock_logger.statuses
        ), mock_logger.statuses


# ---------------------------------------------------------------------------
# TestOpenFirstCapture (CAP-02, CAP-06, TEST-03)
# ---------------------------------------------------------------------------

class TestOpenFirstCapture:
    """CAP-02: first call on open|datagen captures the image at
    {results_dir}/open/<orgname>/results/<systemname>/<benchmark>/<model>/code/.
    """

    def test_open_first_capture_creates_per_leaf_code_dir(
        self, tmp_path, fake_source_root, mock_logger
    ):
        args = make_args(
            mode="open", command="datagen", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "sys-1"}
        result = capture_or_verify_code_image(args, env, mock_logger)
        expected = (
            tmp_path / "open" / "acme" / "results" / "sys-1"
            / "training" / "unet3d" / "code"
        )
        assert result == expected
        assert expected.is_dir()
        assert (expected / ".code-hash.json").is_file()

    def test_open_first_capture_logs_absolute_path(
        self, tmp_path, fake_source_root, mock_logger
    ):
        args = make_args(
            mode="open", command="datagen", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "sys-1"}
        capture_or_verify_code_image(args, env, mock_logger)
        expected = (
            tmp_path / "open" / "acme" / "results" / "sys-1"
            / "training" / "unet3d" / "code"
        )
        assert any(
            s.startswith("Captured code image at ") and str(expected) in s
            for s in mock_logger.statuses
        ), mock_logger.statuses


# ---------------------------------------------------------------------------
# TestRuntimeMatchPasses (VALR-01, VALR-03, TEST-04)
# ---------------------------------------------------------------------------

class TestRuntimeMatchPasses:
    """VALR-01/03: second call against an unchanged tree logs the
    'code unchanged from on-file image at <path>' status and returns the path.
    """

    def test_closed_second_run_matches(
        self, tmp_path, fake_source_root, mock_logger
    ):
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        # First call captures
        first = capture_or_verify_code_image(args, env, mock_logger)
        mock_logger.statuses.clear()
        # Second call must verify silently
        second = capture_or_verify_code_image(args, env, mock_logger)
        assert second == first
        expected = tmp_path / "closed" / "acme" / "code"
        assert any(
            f"code unchanged from on-file image at {expected}" in s
            for s in mock_logger.statuses
        ), mock_logger.statuses

    def test_open_second_run_matches(
        self, tmp_path, fake_source_root, mock_logger
    ):
        args = make_args(
            mode="open", command="datagen", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "sys-1"}
        first = capture_or_verify_code_image(args, env, mock_logger)
        mock_logger.statuses.clear()
        second = capture_or_verify_code_image(args, env, mock_logger)
        assert second == first
        expected = (
            tmp_path / "open" / "acme" / "results" / "sys-1"
            / "training" / "unet3d" / "code"
        )
        assert any(
            f"code unchanged from on-file image at {expected}" in s
            for s in mock_logger.statuses
        ), mock_logger.statuses


# ---------------------------------------------------------------------------
# TestRuntimeMismatchCLOSED (VALR-02, TEST-05)
# ---------------------------------------------------------------------------

class TestRuntimeMismatchCLOSED:
    """VALR-02: on hash mismatch in a CLOSED run, raise CodeImageError
    containing the literal spec string
    'changes to the codebase are not allowed in a CLOSED run'.
    """

    def test_closed_mismatch_raises_with_literal_message(
        self, tmp_path, fake_source_root, mock_logger, monkeypatch
    ):
        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        # First call captures successfully.
        capture_or_verify_code_image(args, env, mock_logger)

        # Force a hash mismatch on the second call by monkeypatching
        # verify_source_against_image to return False. This isolates the
        # mismatch code path from the Phase 1 capture-vs-verify hash
        # discrepancy documented in deferred-items.md.
        import mlpstorage_py.submission_checker.tools.code_image as mod
        monkeypatch.setattr(mod, "verify_source_against_image", lambda *a, **k: False)

        mock_logger.errors.clear()
        with pytest.raises(CodeImageError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        # Literal spec string (VALR-02 substring match) — required by
        # deep_work_rules. Assert against BOTH the raised exception and the
        # logger so a future regression that drops one path still fails.
        assert "changes to the codebase are not allowed in a CLOSED run" in str(exc_info.value)
        assert any(
            "changes to the codebase are not allowed in a CLOSED run" in e
            for e in mock_logger.errors
        ), mock_logger.errors
        code_dir = tmp_path / "closed" / "acme" / "code"
        assert any(f"code image at: {code_dir}" in e for e in mock_logger.errors), mock_logger.errors


# ---------------------------------------------------------------------------
# TestRuntimeMismatchOPEN (VALR-04, TEST-06)
# ---------------------------------------------------------------------------

class TestRuntimeMismatchOPEN:
    """VALR-04: on hash mismatch in an OPEN run, raise CodeImageError
    containing the literal spec string
    'all runs of this type must use the same codebase'.
    """

    def test_open_mismatch_raises_with_literal_message(
        self, tmp_path, fake_source_root, mock_logger, monkeypatch
    ):
        args = make_args(
            mode="open", command="datagen", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "sys-1"}
        capture_or_verify_code_image(args, env, mock_logger)

        import mlpstorage_py.submission_checker.tools.code_image as mod
        monkeypatch.setattr(mod, "verify_source_against_image", lambda *a, **k: False)

        mock_logger.errors.clear()
        with pytest.raises(CodeImageError) as exc_info:
            capture_or_verify_code_image(args, env, mock_logger)
        assert "all runs of this type must use the same codebase" in str(exc_info.value)
        assert any(
            "all runs of this type must use the same codebase" in e
            for e in mock_logger.errors
        ), mock_logger.errors
        code_dir = (
            tmp_path / "open" / "acme" / "results" / "sys-1"
            / "training" / "unet3d" / "code"
        )
        assert any(f"code image at: {code_dir}" in e for e in mock_logger.errors), mock_logger.errors


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
# TestBadImageRecovery (D-21)
# ---------------------------------------------------------------------------

class TestBadImageRecovery:
    """D-21: when an existing code/ has a missing or malformed .code-hash.json,
    the helper logs the actionable recovery substring and re-raises the
    Phase 1 typed error.
    """

    def test_missing_hash_file_logs_recovery_message(self, tmp_path, mock_logger):
        # Pre-create code/ with files but NO .code-hash.json.
        code_dir = tmp_path / "closed" / "acme" / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "dummy.py").write_text("# placeholder\n")

        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(MissingHashFile):
            capture_or_verify_code_image(args, env, mock_logger)
        assert any(
            "either delete `code/` and re-run to re-capture, or restore the original capture."
            in e
            for e in mock_logger.errors
        ), mock_logger.errors

    def test_malformed_hash_file_logs_recovery_message(self, tmp_path, mock_logger):
        # Pre-create code/ with an invalid .code-hash.json.
        code_dir = tmp_path / "closed" / "acme" / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "dummy.py").write_text("# placeholder\n")
        (code_dir / ".code-hash.json").write_text("{not valid json")

        args = make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(MalformedHashFile):
            capture_or_verify_code_image(args, env, mock_logger)
        assert any(
            "either delete `code/` and re-run to re-capture, or restore the original capture."
            in e
            for e in mock_logger.errors
        ), mock_logger.errors
