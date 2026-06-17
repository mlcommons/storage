"""Tests for BUG-T4: 2.1.24 checkpointingTimestampGap — sentinel-type mismatch.

The original implementation initialised the "shortest run duration" tracker
as ``max_gap = float("inf")`` and then compared ``run_duration < max_gap``
on the first iteration. ``run_duration`` is a ``datetime.timedelta``, so
Python 3.10+ raises ``TypeError: '<' not supported between instances of
'datetime.timedelta' and 'float'`` on every checkpoint timestamp. The
existing ``except (ValueError, KeyError, TypeError)`` swallows the
TypeError and logs it as a 2.1.24 violation — masking the real bug
behind a misleading "Failed to parse timestamp data" diagnostic on
otherwise-valid submissions.

Fix: use ``timedelta.max`` as the sentinel so the running minimum stays
a timedelta. Also guard against None summary dicts (missing summary.json
is already reported under 2.1.22).
"""

from unittest.mock import MagicMock

from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


def _make_directory_check(tmp_path, checkpoint_files):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        checkpoint_files=checkpoint_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode="checkpointing",
            benchmark="llama3-8b",
            folder=str(tmp_path),
        ),
    )
    return DirectoryCheck(log=log, config=config, submissions_logs=submissions_logs)


def test_bug_t4_valid_checkpoints_do_not_log_2_1_24_parse_failure(tmp_path):
    """Well-formed checkpoint timestamps must not produce a 2.1.24 parse-failure
    violation. Pre-fix: TypeError on float("inf") < timedelta → spurious violation.
    Post-fix: no 2.1.24 violation fires on valid input.
    """
    checkpoint_files = [
        (
            {"start": "2025-07-11T19:50:50", "end": "2025-07-11T19:54:25"},
            {},
            "20250711_195047",
        ),
        (
            {"start": "2025-07-11T19:52:50", "end": "2025-07-11T19:56:25"},
            {},
            "20250711_195247",
        ),
    ]
    check = _make_directory_check(tmp_path, checkpoint_files)
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True, (
        "2.1.24 should not fire on well-formed checkpoint timestamps; "
        "this indicates the timedelta/float sentinel mismatch resurfaced"
    )


def test_bug_t4_none_checkpoint_dict_is_skipped(tmp_path):
    """Missing summary.json (checkpoint_dict is None) must not crash 2.1.24.

    Already reported under 2.1.22 (checkpointingResultsJson) by the
    structural check; this method should silently skip.
    """
    checkpoint_files = [
        (None, None, "20250711_195047"),
        (
            {"start": "2025-07-11T19:52:50", "end": "2025-07-11T19:56:25"},
            {},
            "20250711_195247",
        ),
    ]
    check = _make_directory_check(tmp_path, checkpoint_files)
    # Must not raise; valid because nothing actually flags 2.1.24.
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True


def test_bug_t4_single_checkpoint_does_not_crash(tmp_path):
    """Single checkpoint timestamp pre-fix raised TypeError on the sentinel
    comparison even though there's no pair to gap-check."""
    checkpoint_files = [
        (
            {"start": "2025-07-11T19:50:50", "end": "2025-07-11T19:54:25"},
            {},
            "20250711_195047",
        ),
    ]
    check = _make_directory_check(tmp_path, checkpoint_files)
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True
