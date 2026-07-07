#!/usr/bin/env python3
"""Phase 6 Plan 06-02 Task 1 (RED) — D-63 LegacyLayoutDetected refusal tests.

Locks the D-63 refusal semantics for `capture_or_verify_code_image`: if a
legacy `code/` directory is present under `<results_dir>/{closed,open}/<orgname>/`,
the helper must raise `LegacyLayoutDetected` (a `CodeImageError` subclass)
BEFORE any capture-side writes touch the pool. The refusal preserves the
strict single-layout invariant that Phase 8's CHECK-04 assumes.

Fixtures (`MockLogger`, `_make_args`) are duplicated verbatim from
`mlpstorage_py/tests/test_capture_or_verify_code_image.py:33-75` per PATTERNS
`### New unit test files under mlpstorage_py/tests/` guidance.

Run with:
    pytest mlpstorage_py/tests/test_legacy_layout_refuse.py -v
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,
    LegacyLayoutDetected,
    capture_or_verify_code_image,
)


# ---------------------------------------------------------------------------
# MockLogger — same shape as the analog (test_capture_or_verify_code_image.py:33-64)
# ---------------------------------------------------------------------------

class MockLogger:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.warnings = []
        self.infos = []
        self.debugs = []

    def status(self, msg, *args):  self.statuses.append(msg % args if args else msg)
    def error(self, msg, *args):   self.errors.append(msg % args if args else msg)
    def warning(self, msg, *args): self.warnings.append(msg % args if args else msg)
    def info(self, msg, *args):    self.infos.append(msg % args if args else msg)
    def debug(self, msg, *args):   self.debugs.append(msg % args if args else msg)
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def log():
    return MockLogger()


def _make_args(
    *,
    mode,
    command,
    results_dir,
    benchmark="training",
    model="unet3d",
    orgname=None,
    systemname=None,
):
    return SimpleNamespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
        orgname=orgname,
        systemname=systemname,
    )


# ---------------------------------------------------------------------------
# Class-scoped tests: D-63 legacy layout refusal
# ---------------------------------------------------------------------------

class TestLegacyLayoutRefuse:
    """D-63: refuse pool-image writes when a legacy `code/` directory is
    present. Message names the offender path and includes the Phase 7
    auto-migrate hint.
    """

    def test_closed_mode_legacy_code_dir_raises(self, tmp_path, log):
        results_dir = tmp_path / "results"
        legacy = results_dir / "closed" / "Acme" / "code"
        legacy.mkdir(parents=True)

        args = _make_args(
            mode="closed",
            command="run",
            results_dir=results_dir,
            orgname="Acme",
        )
        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        assert "Legacy code-image layout detected" in str(exc_info.value)
        # Path substring should be present (offender named in message).
        assert str(legacy) in str(exc_info.value) or repr(legacy) in str(exc_info.value)

    def test_open_mode_legacy_code_dir_raises(self, tmp_path, log):
        results_dir = tmp_path / "results"
        legacy = results_dir / "open" / "Acme" / "code"
        legacy.mkdir(parents=True)

        args = _make_args(
            mode="open",
            command="run",
            results_dir=results_dir,
            benchmark="training",
            model="unet3d",
            orgname="Acme",
            systemname="rig01",
        )
        env = {"MLPSTORAGE_SYSTEMNAME": "rig01"}
        with pytest.raises(LegacyLayoutDetected):
            capture_or_verify_code_image(args, env, log)

    def test_open_mode_deep_legacy_code_dir_still_caught_via_parent_check(
        self, tmp_path, log
    ):
        """The legacy open path is
        `results_dir/open/<org>/code/<benchmark>/<command>/`, so the parent
        `results_dir/open/<org>/code/` alone is what `_scan_legacy_layout`
        checks. Any deep tree under that parent must still be caught by the
        one-syscall-per-mode scan.
        """
        results_dir = tmp_path / "results"
        deep = results_dir / "open" / "Acme" / "code" / "training" / "run"
        deep.mkdir(parents=True)

        args = _make_args(
            mode="open",
            command="run",
            results_dir=results_dir,
            benchmark="training",
            model="unet3d",
            orgname="Acme",
            systemname="rig01",
        )
        env = {"MLPSTORAGE_SYSTEMNAME": "rig01"}
        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, env, log)
        # Parent path `open/Acme/code` should appear in the message.
        parent = results_dir / "open" / "Acme" / "code"
        assert str(parent) in str(exc_info.value) or repr(parent) in str(exc_info.value)

    def test_both_closed_and_open_legacy_present_message_mentions_first_and_count(
        self, tmp_path, log
    ):
        """D-63 spec: name the first offender; report a count of extras."""
        results_dir = tmp_path / "results"
        (results_dir / "closed" / "Acme" / "code").mkdir(parents=True)
        (results_dir / "open" / "Acme" / "code").mkdir(parents=True)

        args = _make_args(
            mode="closed",
            command="run",
            results_dir=results_dir,
            orgname="Acme",
        )
        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        msg = str(exc_info.value)
        # "+1 more" for the sibling offender.
        assert "+1 more" in msg or "1 more" in msg

    def test_LegacyLayoutDetected_is_a_CodeImageError(self):
        """main.py's exit-code mapping catches CodeImageError. Subclassing
        ensures LegacyLayoutDetected is caught by the same handler."""
        assert issubclass(LegacyLayoutDetected, CodeImageError)

    def test_LegacyLayoutDetected_message_contains_phase_7_migration_hint(
        self, tmp_path, log
    ):
        results_dir = tmp_path / "results"
        (results_dir / "closed" / "Acme" / "code").mkdir(parents=True)

        args = _make_args(
            mode="closed",
            command="run",
            results_dir=results_dir,
            orgname="Acme",
        )
        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        assert "auto-migrate on your next submission-mode run" in str(exc_info.value)

    def test_no_legacy_no_refusal(self, tmp_path, log):
        """No `code/` under `results/{closed,open}/Acme/` → the scan MUST
        NOT raise LegacyLayoutDetected. The function may still fail for
        other reasons (pool infra not fully wired at RED stage) — we only
        assert LegacyLayoutDetected specifically is not raised.
        """
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed",
            command="run",
            results_dir=results_dir,
            orgname="Acme",
        )
        try:
            capture_or_verify_code_image(args, {}, log)
        except LegacyLayoutDetected:
            pytest.fail("LegacyLayoutDetected raised when no legacy `code/` present")
        except Exception:
            # Other exceptions permitted at RED; only the LegacyLayoutDetected
            # negative-assertion matters here.
            pass

    def test_pool_code_hash8_dir_is_not_flagged_as_legacy(self, tmp_path, log):
        """Pool image dirs (`code-<hash8>`) are NOT the literal name `code`
        so `_scan_legacy_layout` skips them. Pre-seed a valid pool image
        shape and assert the scan does not refuse.
        """
        results_dir = tmp_path / "results"
        # Pre-seed the pool dir; content is irrelevant here — this test only
        # asserts the legacy-scan does not flag the `code-<suffix>` name.
        pool = results_dir / "Acme" / "code-abcd1234"
        pool.mkdir(parents=True)
        # A minimal .code-hash.json so any downstream helper that reads it
        # can skip cleanly at DEBUG (the scan itself only checks the name).
        (pool / ".code-hash.json").write_text('{"hash": "abcd1234"}\n')

        args = _make_args(
            mode="closed",
            command="run",
            results_dir=results_dir,
            orgname="Acme",
        )
        try:
            capture_or_verify_code_image(args, {}, log)
        except LegacyLayoutDetected:
            pytest.fail(
                "LegacyLayoutDetected raised for a code-<hash8> pool dir "
                "(not the literal name `code`)"
            )
        except Exception:
            # Other errors are acceptable at RED; only LegacyLayoutDetected
            # is what this test negates.
            pass
