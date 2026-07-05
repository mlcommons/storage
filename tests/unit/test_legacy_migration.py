"""Wave-0 xfail scaffolding for legacy_migration.py unit tests.

Isolated unit tests (mocked FS, no real disk I/O) targeting the functions
introduced in Plan 07-02's ``mlpstorage_py/submission_checker/tools/legacy_migration.py``:

  - _verify_all_legacy_dirs: pass-1 hash-verify loop (raises HandEditedCodeImage)
  - _write_sentinel_atomic: atomic sentinel writer (write-tmp + os.rename)
  - _read_sentinel: plain-text key=value reader (ignores unknown fields)
  - _enumerate_run_leaves: bounded glob that discovers datetime dirs across all 3 shapes
  - _check_and_migrate_legacy_layout: pre-check helper called from main.py (D-70)
  - HandEditedCodeImage: exception subclassing CodeImageError (D-73)

Wave 0 note: every test stub raises NotImplementedError and is marked
xfail(strict=True). Wave-2 (Plan 07-04) removes xfail decorators and
populates test bodies. No production symbols are imported here — they do not
exist until Plan 07-02. All references to production functions are in docstrings
only.

Refs: 07-01-PLAN.md Task 3, 07-CONTEXT.md D-70/D-71/D-72/D-73,
RESEARCH §10 "Unit tests" table, PATTERNS §test_legacy_migration.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock


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

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_hash_mismatch_raises_HandEditedCodeImage_with_first_offender_and_plus_N_more(
        self,
    ):
        """_verify_all_legacy_dirs raises HandEditedCodeImage on hash mismatch.

        Given two legacy dirs where the first re-hashes differently from its
        stored .code-hash.json digest, assert that HandEditedCodeImage is raised
        with a message naming the first offender and including "+N more" hint
        (D-73 error format). The second legacy dir is not checked after the first
        mismatch (fail-fast per pass-1 spec).
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )


class TestSentinelWriter:
    """Unit tests for _write_sentinel_atomic (D-72 format + D-65 atomic write)."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_writes_two_named_lines_with_trailing_newline_atomically(self):
        """_write_sentinel_atomic writes exactly two key=value lines + trailing newline.

        D-72 format: ``mlpstorage_version=<semver>\nmigration_completed_at=<ISO>\n``.
        Assert:
        - Sentinel file content matches this pattern.
        - Write is atomic: no tmp sibling file remains after write.
        - The tmp file path is dot-prefixed (D-65 convention).
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )


class TestSentinelReader:
    """Unit tests for _read_sentinel (D-72 forward-compatible reader)."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_ignores_unknown_key_equals_value_lines(self):
        """_read_sentinel ignores unknown key=value lines for forward compatibility.

        D-72: reader is forward-compatible — unknown keys added in future versions
        are silently ignored. Assert that a sentinel with an extra ``foo=bar`` line
        returns the known fields correctly and does not raise.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )


class TestEnumerateRunLeaves:
    """Unit tests for _enumerate_run_leaves across all three benchmark shapes."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_finds_training_shape_5_level(self):
        """_enumerate_run_leaves discovers datetime dirs in training (5-level) shape.

        5-level shape: results/<sys>/<bench>/<model>/<cmd>/<datetime>/.
        Given a tmp tree with N training datetime dirs, assert _enumerate_run_leaves
        returns exactly N paths, each ending in a datetime-format dir name.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_finds_checkpointing_shape_4_level(self):
        """_enumerate_run_leaves discovers datetime dirs in checkpointing (4-level) shape.

        4-level shape: results/<sys>/<bench>/<model>/<datetime>/ (no command level).
        Assert N leaves discovered correctly.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_finds_vector_database_shape_6_level(self):
        """_enumerate_run_leaves discovers datetime dirs in vector_database (6-level) shape.

        6-level shape: results/<sys>/<bench>/<engine>/<index>/<cmd>/<datetime>/.
        Assert N leaves discovered correctly.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )


class TestPreCheckHelper:
    """Unit tests for _check_and_migrate_legacy_layout (D-70 pre-check in main.py)."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_sentinel_present_skips_scan(self):
        """_check_and_migrate_legacy_layout skips scan when sentinel already present.

        If <results_dir>/<orgname>/.mlps-image-pool exists, the helper must
        return immediately without calling _scan_legacy_layout (O(2) syscall
        fast path — D-70). Assert via monkeypatch spy that _scan_legacy_layout
        call_count == 0.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_non_submission_mode_no_ops(self):
        """_check_and_migrate_legacy_layout is a no-op for non-submission commands.

        Commands like datagen and datasize do not trigger migration — only
        submission-scoped commands (run, configview) where a results_dir is
        meaningful trigger the pre-check. Assert that calling the helper with
        a non-submission command leaves the filesystem unchanged.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )


class TestHandEditedCodeImageSubclass:
    """Unit tests for the HandEditedCodeImage exception hierarchy."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
    def test_subclasses_CodeImageError(self):
        """HandEditedCodeImage subclasses CodeImageError for main.py exit-code mapping.

        D-73: HandEditedCodeImage must be a subclass of CodeImageError so the
        existing exit-code mapping in main.py catches it without a new handler.
        Assert issubclass(HandEditedCodeImage, CodeImageError) is True.
        """
        raise NotImplementedError(
            "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
        )
