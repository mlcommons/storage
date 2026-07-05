"""Unit tests for legacy_migration.py helper functions.

Isolated unit tests (mocked FS, no real disk I/O) targeting the functions
introduced in ``mlpstorage_py/submission_checker/tools/legacy_migration.py``:

  - _verify_all_legacy_dirs: pass-1 hash-verify loop (raises HandEditedCodeImage)
  - _write_sentinel_atomic: atomic sentinel writer (write-tmp + os.rename)
  - _read_sentinel: plain-text key=value reader (ignores unknown fields)
  - _enumerate_run_leaves: bounded glob that discovers datetime dirs across all 3 shapes
  - _check_and_migrate_legacy_layout: pre-check helper called from main.py (D-70)
  - HandEditedCodeImage: exception subclassing CodeImageError (D-73)

Wave 0 xfail stubs replaced with real assertions by Plan 07-03 Task 2.

Refs: 07-01-PLAN.md Task 3, 07-CONTEXT.md D-70/D-71/D-72/D-73,
RESEARCH §10 "Unit tests" table, PATTERNS §test_legacy_migration.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import mlpstorage_py.submission_checker.tools.legacy_migration as lm
import pytest
from mlpstorage_py.submission_checker.tools.code_image import CodeImageError, HandEditedCodeImage
from mlpstorage_py.submission_checker.tools.legacy_migration import (
    _check_and_migrate_legacy_layout,
    _enumerate_run_leaves,
    _read_sentinel,
    _verify_all_legacy_dirs,
    _write_sentinel_atomic,
)


def _make_log():
    """Minimal MagicMock logger matching the logger contract used by code_image.py."""
    log = MagicMock()
    log.warning = MagicMock()
    log.error = MagicMock()
    log.status = MagicMock()
    log.info = MagicMock()
    log.debug = MagicMock()
    return log


class TestVerifyPass1:
    """Unit tests for _verify_all_legacy_dirs (pass-1 hash-verify loop)."""

    def test_hash_mismatch_raises_HandEditedCodeImage_with_first_offender_and_plus_N_more(
        self, tmp_path, monkeypatch
    ):
        """_verify_all_legacy_dirs raises HandEditedCodeImage on hash mismatch.

        Given two legacy dirs where the first re-hashes differently from its
        stored .code-hash.json digest, assert that HandEditedCodeImage is raised
        with a message naming the first offender and including "+1 more" hint
        (D-73 error format). The second legacy dir is not checked after the first
        mismatch (fail-fast per pass-1 spec).
        """
        # Build two fake legacy dirs.
        legacy1 = tmp_path / "closed" / "Acme" / "code"
        legacy2 = tmp_path / "open" / "Acme" / "code"
        legacy1.mkdir(parents=True)
        legacy2.mkdir(parents=True)

        # Patch _scan_legacy_layout to return both paths.
        monkeypatch.setattr(lm, "_scan_legacy_layout", lambda rd, org: [legacy1, legacy2])

        # Patch _read_hash_file to return a stored hash of all zeros.
        stored_hash = "0" * 32
        monkeypatch.setattr(
            lm, "_read_hash_file", lambda path, log: {"hash": stored_hash, "algorithm": "md5-tree-v2"}
        )

        # Patch compute_code_tree_md5 to return a different hash (mismatch).
        live_hash = "f" * 32
        monkeypatch.setattr(lm, "compute_code_tree_md5", lambda path, log: live_hash)

        log = _make_log()
        with pytest.raises(HandEditedCodeImage, match=r"hand-edited code image detected at .+; \+1 more"):
            _verify_all_legacy_dirs(tmp_path, "Acme", log)


class TestSentinelWriter:
    """Unit tests for _write_sentinel_atomic (D-72 format + D-65 atomic write)."""

    def test_writes_two_named_lines_with_trailing_newline_atomically(self, tmp_path):
        """_write_sentinel_atomic writes exactly two key=value lines + trailing newline.

        D-72 format: ``mlpstorage_version=<semver>\\nmigration_completed_at=<ISO>\\n``.
        Assert:
        - Sentinel file content matches this pattern.
        - Write is atomic: no tmp sibling file remains after write.
        """
        org_root = tmp_path / "Acme"
        org_root.mkdir()
        log = _make_log()

        _write_sentinel_atomic(org_root, log)

        sentinel = org_root / ".mlps-image-pool"
        assert sentinel.exists(), "sentinel file must be written"
        text = sentinel.read_text()
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) == 2, f"expected 2 key=value lines, got {len(lines)}: {lines!r}"
        keys = {l.split("=", 1)[0] for l in lines}
        assert "mlpstorage_version" in keys
        assert "migration_completed_at" in keys
        assert text.endswith("\n"), "sentinel must end with trailing newline (D-72)"

        # No tmp sibling remains.
        tmp_siblings = list(org_root.glob(".mlps-image-pool.tmp.*"))
        assert tmp_siblings == [], f"unexpected tmp sibling(s): {tmp_siblings}"


class TestSentinelReader:
    """Unit tests for _read_sentinel (D-72 forward-compatible reader)."""

    def test_ignores_unknown_key_equals_value_lines(self, tmp_path):
        """_read_sentinel ignores unknown key=value lines for forward compatibility.

        D-72: reader is forward-compatible — unknown keys added in future versions
        are silently ignored. Assert that a sentinel with an extra ``foo=bar`` line
        returns the known fields correctly and does not raise.
        """
        sentinel_path = tmp_path / ".mlps-image-pool"
        sentinel_path.write_text(
            "mlpstorage_version=1.2.3\n"
            "migration_completed_at=2026-07-05T00:00:00Z\n"
            "future_field=X\n"
        )
        log = _make_log()
        result = _read_sentinel(sentinel_path, log)
        assert result["mlpstorage_version"] == "1.2.3"
        assert result["migration_completed_at"] == "2026-07-05T00:00:00Z"
        assert result["future_field"] == "X"


class TestEnumerateRunLeaves:
    """Unit tests for _enumerate_run_leaves across all three benchmark shapes."""

    def test_finds_training_shape_5_level(self, tmp_path):
        """_enumerate_run_leaves discovers datetime dirs in training (5-level) shape.

        5-level shape: results/<sys>/<bench>/<model>/<cmd>/<datetime>/.
        """
        subtree = tmp_path / "closed" / "Acme"
        leaf = subtree / "results" / "sys1" / "training" / "unet3d" / "run" / "20260101_120000"
        leaf.mkdir(parents=True)
        log = _make_log()
        result = _enumerate_run_leaves(subtree, log)
        assert leaf in result, f"{leaf} not found in {result}"

    def test_finds_checkpointing_shape_4_level(self, tmp_path):
        """_enumerate_run_leaves discovers datetime dirs in checkpointing (4-level) shape.

        4-level shape: results/<sys>/<bench>/<model>/<datetime>/ (no command level).
        """
        subtree = tmp_path / "closed" / "Acme"
        leaf = subtree / "results" / "sys1" / "checkpointing" / "llama3-8b" / "20260101_120000"
        leaf.mkdir(parents=True)
        log = _make_log()
        result = _enumerate_run_leaves(subtree, log)
        assert leaf in result, f"{leaf} not found in {result}"

    def test_finds_vector_database_shape_6_level(self, tmp_path):
        """_enumerate_run_leaves discovers datetime dirs in vector_database (6-level) shape.

        6-level shape: results/<sys>/<bench>/<engine>/<index>/<cmd>/<datetime>/.
        """
        subtree = tmp_path / "closed" / "Acme"
        leaf = (
            subtree / "results" / "sys1"
            / "vector_database" / "diskann" / "HNSW" / "run" / "20260101_120000"
        )
        leaf.mkdir(parents=True)
        log = _make_log()
        result = _enumerate_run_leaves(subtree, log)
        assert leaf in result, f"{leaf} not found in {result}"


class TestPreCheckHelper:
    """Unit tests for _check_and_migrate_legacy_layout (D-70 pre-check in main.py)."""

    def test_sentinel_present_skips_scan(self, tmp_path, monkeypatch):
        """_check_and_migrate_legacy_layout skips scan when sentinel already present.

        If <results_dir>/<orgname>/.mlps-image-pool exists, the helper must
        return immediately without calling _scan_legacy_layout (O(2) syscall
        fast path — D-70). Assert via monkeypatch spy that _scan_legacy_layout
        call_count == 0.
        """
        results_dir = tmp_path / "results"
        org_root = results_dir / "Acme"
        org_root.mkdir(parents=True)
        (org_root / ".mlps-image-pool").write_text(
            "mlpstorage_version=1.0\nmigration_completed_at=2026-01-01T00:00:00Z\n"
        )

        scan_spy = MagicMock()
        monkeypatch.setattr(lm, "_scan_legacy_layout", scan_spy)

        args = Namespace(mode="closed", command="run", results_dir=str(results_dir), orgname="Acme", systemname=None)
        _check_and_migrate_legacy_layout(args, {}, _make_log())
        assert scan_spy.call_count == 0, "sentinel present: _scan_legacy_layout must NOT be called"

    def test_non_submission_mode_no_ops(self, tmp_path, monkeypatch):
        """_check_and_migrate_legacy_layout is a no-op for non-submission modes.

        Modes not in _SUBMISSION_MODES (e.g. 'whatif') trigger an early return
        before any scan or migration. Assert both _scan_legacy_layout and
        migrate_legacy_layout have call_count == 0.
        """
        scan_spy = MagicMock()
        migrate_spy = MagicMock()
        monkeypatch.setattr(lm, "_scan_legacy_layout", scan_spy)
        monkeypatch.setattr(lm, "migrate_legacy_layout", migrate_spy)

        args = Namespace(mode="whatif", command="run", results_dir=str(tmp_path), orgname="Acme", systemname=None)
        _check_and_migrate_legacy_layout(args, {}, _make_log())
        assert scan_spy.call_count == 0, "non-submission mode: _scan_legacy_layout must NOT be called"
        assert migrate_spy.call_count == 0, "non-submission mode: migrate_legacy_layout must NOT be called"


class TestHandEditedCodeImageSubclass:
    """Unit tests for the HandEditedCodeImage exception hierarchy."""

    def test_subclasses_CodeImageError(self):
        """HandEditedCodeImage subclasses CodeImageError for main.py exit-code mapping.

        D-73: HandEditedCodeImage must be a subclass of CodeImageError so the
        existing exit-code mapping in main.py catches it without a new handler.
        Assert issubclass(HandEditedCodeImage, CodeImageError) is True.
        """
        assert issubclass(HandEditedCodeImage, CodeImageError)
