#!/usr/bin/env python3
"""
Regression tests for BUG-04: DirectoryCheck.run_files_timestamp_check must
require exactly RUN_TIMESTAMP_COUNT (=6) run-file timestamps — 1 warm-up + 5
measured — per Rules.md 2.1.17 (runTimestamps).

Pins:
  * The stale `# v2.0 only 5 required` comment is gone.
  * The inline literal `6` has been replaced with the named constant
    `RUN_TIMESTAMP_COUNT` (so the magic-number anti-pattern PITFALLS.md #8
    flagged can't regress).
  * The docstring cites Rules.md 2.1.17 and the warm-up + measured semantic.
  * The invalid-timestamp-format check still flags malformed entries.

Run with:
    pytest mlpstorage_py/tests/test_directory_check_run_timestamps.py -v
"""

import inspect
from unittest.mock import MagicMock

import pytest

import mlpstorage_py.submission_checker.checks.directory_checks as dc_mod
from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
from mlpstorage_py.submission_checker.constants import RUN_TIMESTAMP_COUNT


class CapturingLogger:
    """Mock logger that captures error/warning calls for assertion.

    Stores both the format string and the post-`%`-formatted message so tests
    can assert on either form.
    """

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []

    def _record(self, bucket, msg, args):
        try:
            formatted = msg % args if args else msg
        except (TypeError, ValueError):
            formatted = msg
        bucket.append(formatted)

    def debug(self, msg, *args): pass
    def info(self, msg, *args): self._record(self.infos, msg, args)
    def warning(self, msg, *args): self._record(self.warnings, msg, args)
    def error(self, msg, *args): self._record(self.errors, msg, args)
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def mock_logger():
    return CapturingLogger()


def _make_check(timestamps, log):
    """Build a DirectoryCheck without invoking __init__ (which requires a
    real SubmissionLogs + filesystem path). Set only the attributes
    run_files_timestamp_check actually touches: self.log,
    self.submissions_logs.run_files, and self.run_path (used by
    log_violation after the Phase 3 retrofit).
    """
    check = DirectoryCheck.__new__(DirectoryCheck)
    check.log = log
    check.run_path = "/test/run"
    submissions_logs = MagicMock()
    submissions_logs.run_files = [
        (f"run_file_{i}", f"metadata_{i}", ts) for i, ts in enumerate(timestamps)
    ]
    check.submissions_logs = submissions_logs
    return check


def _ts(n):
    """Return n valid YYYYMMDD_HHmmss timestamp strings."""
    base_minute = 0
    out = []
    for i in range(n):
        out.append(f"20260101_12{base_minute + i:04d}")
    return out


class TestRunFilesTimestampCount:
    """BUG-04 count-fix regression."""

    def test_five_timestamps_fails_and_logs_six(self, mock_logger):
        """5 valid timestamps must fail and the error must mention both 6 (expected) and 5 (actual)."""
        check = _make_check(_ts(5), mock_logger)
        result = check.run_files_timestamp_check()
        assert result is False, "5 timestamps must NOT pass the count check"
        assert mock_logger.errors, "expected an error to be logged for wrong count"
        joined = " ".join(mock_logger.errors)
        assert "6" in joined, f"expected '6' (RUN_TIMESTAMP_COUNT) in error; got: {joined!r}"
        assert "5" in joined, f"expected '5' (actual count) in error; got: {joined!r}"

    def test_six_timestamps_passes_with_no_error(self, mock_logger):
        """6 valid timestamps must pass and log no errors."""
        check = _make_check(_ts(6), mock_logger)
        result = check.run_files_timestamp_check()
        assert result is True, "6 timestamps MUST pass the count check"
        assert mock_logger.errors == [], (
            f"expected no errors for the happy path; got: {mock_logger.errors!r}"
        )

    def test_count_is_driven_by_constant_not_inline_literal(self, monkeypatch, mock_logger):
        """Monkeypatching RUN_TIMESTAMP_COUNT in the directory_checks module must
        change the accepted count. If the check still used inline `6`, this would
        fail — proving the magic-number anti-pattern is gone.
        """
        monkeypatch.setattr(dc_mod, "RUN_TIMESTAMP_COUNT", 7)
        check = _make_check(_ts(7), mock_logger)
        result = check.run_files_timestamp_check()
        assert result is True, (
            "with RUN_TIMESTAMP_COUNT monkeypatched to 7, a 7-timestamp fixture "
            "must pass — proving the inline `6` was removed"
        )
        assert mock_logger.errors == [], (
            f"expected no errors when count matches patched constant; got: {mock_logger.errors!r}"
        )


class TestRunFilesTimestampSource:
    """Source-level assertions pinning the BUG-04 intent (no stale comment,
    docstring cites the rule)."""

    def test_stale_comment_is_gone_and_rule_cited(self):
        src = inspect.getsource(DirectoryCheck.run_files_timestamp_check)
        assert "v2.0 only 5 required" not in src, (
            "stale `# v2.0 only 5 required` comment must be deleted (BUG-04)"
        )
        assert "Rules.md 2.1.17" in src, (
            "docstring must cite Rules.md 2.1.17 (runTimestamps) so submitters "
            "can grep the spec from validator output"
        )


class TestRunFilesTimestampFormat:
    """The format regex check must still flag malformed entries — BUG-04's
    count fix must not regress the adjacent format check."""

    def test_invalid_format_still_flagged(self, mock_logger):
        timestamps = _ts(5) + ["not-a-timestamp"]  # 6 total, one bad
        check = _make_check(timestamps, mock_logger)
        result = check.run_files_timestamp_check()
        assert result is False, "an invalid timestamp format must fail the check"
        joined = " ".join(mock_logger.errors)
        assert "Invalid timestamp format" in joined, (
            f"expected 'Invalid timestamp format' error; got: {mock_logger.errors!r}"
        )


def test_run_timestamp_count_constant_value():
    """Sanity: the constant Plan 02 added is 6 (1 warm-up + 5 measured)."""
    assert RUN_TIMESTAMP_COUNT == 6
