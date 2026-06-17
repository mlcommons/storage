"""Tests for BUG-T3: 2.1.18 runTimestampGap — None summary skip.

When a run timestamp's summary.json is missing, ``Loader.load_single_log``
returns None. The 2.1.18 check then evaluated ``run_dict["start"]`` and
raised ``TypeError: 'NoneType' object is not subscriptable``, which was
swallowed by the broad ``except (ValueError, KeyError, TypeError)`` and
re-emitted as a misleading "Failed to parse timestamp data" violation on
top of the 2.1.19 missing-file diagnostic that already fires for the same
structural complaint.

Fix: skip None entries silently; 2.1.19 owns the diagnostic surface for
the underlying cause.
"""

from unittest.mock import MagicMock

from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


def _make_directory_check(tmp_path, run_files):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode="training",
            benchmark="unet3d",
            folder=str(tmp_path),
        ),
    )
    return DirectoryCheck(log=log, config=config, submissions_logs=submissions_logs)


def test_bug_t3_none_run_dict_is_skipped(tmp_path):
    """A None run_dict must be skipped, not re-reported as 2.1.18 parse failure."""
    run_files = [
        (None, None, "20250711_053436"),
        (
            {"start": "2025-07-11T05:34:36", "end": "2025-07-11T05:38:00"},
            {},
            "20250711_053837",
        ),
    ]
    check = _make_directory_check(tmp_path, run_files)
    log = check.log

    valid = check.run_duration_valid_check()

    # No violation should mention 2.1.18 for the None entry.
    error_calls = [
        c for c in log.error.call_args_list
        if "2.1.18" in str(c) and "20250711_053436" in str(c)
    ]
    assert error_calls == [], (
        "2.1.18 should not double-report the None-summary case; 2.1.19 owns "
        "the missing-file diagnostic. Got: %r" % error_calls
    )
    assert valid is True


def test_bug_t3_all_none_run_dicts_does_not_raise(tmp_path):
    """All-None run_files (every timestamp missing summary.json) must not raise."""
    run_files = [
        (None, None, "20250711_053436"),
        (None, None, "20250711_053837"),
    ]
    check = _make_directory_check(tmp_path, run_files)
    # Must not raise.
    valid = check.run_duration_valid_check()
    assert valid is True
