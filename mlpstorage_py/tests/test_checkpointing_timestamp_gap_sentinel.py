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


# ---------------------------------------------------------------------------
# storage#812 / #813: 2.1.24 must measure the gap from the invocation bookends
# (metadata.invocation_start_time / invocation_end_time), not the DLIO summary
# start/end. The bookends envelope the DLIO loop, so the summary-based gap is
# inflated by read-side framework startup + write-side cluster collection —
# overhead that grows with node count and produced spurious violations on
# otherwise-valid paired submissions (the same class as #714 / #782, which
# fixed the formula but left the source on the summary fields).
# ---------------------------------------------------------------------------


def _paired_invocations_with_inflated_summary():
    """A valid write+read pair where the true quiet window (invocation bookends)
    is 5 s but the DLIO summary window is inflated to a 65 s apparent gap.

    Bookends envelope the DLIO loop: invocation_start < summary.start (framework
    startup) and summary.end < invocation_end (post-benchmark cluster
    collection). Summary durations are 8 s; bookend durations are 40 s.

      bookend gap  = read.invocation_start(19:50:45) - write.invocation_end(19:50:40) = 5 s
                     vs slower bookend duration 40 s  -> PASS
      summary gap  = read.summary.start(19:51:15) - write.summary.end(19:50:10)       = 65 s
                     vs slower summary duration 8 s   -> would FALSE-FAIL (pre-#813)
    """
    write = (
        {"start": "2025-07-11T19:50:02", "end": "2025-07-11T19:50:10"},
        {"invocation_start_time": "2025-07-11T19:50:00",
         "invocation_end_time": "2025-07-11T19:50:40"},
        "20250711_195000",
    )
    read = (
        {"start": "2025-07-11T19:51:15", "end": "2025-07-11T19:51:23"},
        {"invocation_start_time": "2025-07-11T19:50:45",
         "invocation_end_time": "2025-07-11T19:51:25"},
        "20250711_195045",
    )
    return [write, read]


def test_issue_812_gap_uses_invocation_bookends_not_summary(tmp_path):
    """Regression for #812: a paired submission whose real quiet window is small
    must NOT trip 2.1.24 just because the DLIO summary window is inflated by
    startup/collection overhead.

    Pre-#813 this measured read.summary.start - write.summary.end (65 s) against
    the 8 s summary duration and raised a spurious violation. Post-#813 it
    measures the invocation bookends (5 s gap vs 40 s duration) and passes.
    """
    check = _make_directory_check(
        tmp_path, _paired_invocations_with_inflated_summary()
    )
    check.log.error.reset_mock()
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True, (
        "2.1.24 must read the invocation bookends, not the DLIO summary "
        "start/end; the summary window is inflated by read startup + write "
        "cluster collection (storage#812)"
    )
    check.log.error.assert_not_called()


def test_issue_812_bookend_gap_still_flags_a_genuine_long_pause(tmp_path):
    """The #813 source-swap must not neuter the check: a genuinely long quiet
    window measured on the *bookends* must still surface under 2.1.24.

    Worklist A7 (Curtis, 2026-07-24): the gap breach is a WARNING, not an
    error — mirroring the §4.7.1 downgrade in cache_flush_validation /
    check_invocation_structure. The run stays valid but the 5-minute pause
    against 10 s invocations must be reported via warn_violation.
    """
    write = (
        {"start": "2025-07-11T19:49:58", "end": "2025-07-11T19:50:12"},
        {"invocation_start_time": "2025-07-11T19:49:50",
         "invocation_end_time": "2025-07-11T19:50:00"},
        "20250711_194950",
    )
    read = (
        {"start": "2025-07-11T19:54:58", "end": "2025-07-11T19:55:12"},
        {"invocation_start_time": "2025-07-11T19:55:00",
         "invocation_end_time": "2025-07-11T19:55:10"},
        "20250711_195500",
    )
    check = _make_directory_check(tmp_path, [write, read])
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True, (
        "A7: a gap breach warns instead of invalidating — valid must stay True"
    )
    check.log.error.assert_not_called()
    assert check.log.warning.call_count == 1, (
        "the 5-minute bookend gap must still be reported, as a warning"
    )
    warned = check.log.warning.call_args[0][0]
    assert "Gap between checkpoints" in warned


# ---------------------------------------------------------------------------
# Worklist A7 (2026-07-24): 2.1.24 gap breach downgraded to a warning on the
# integration branch (third enforcement point of the §4.7.1 relaxation), and
# the message is labeled as an upper bound when the gap was measured from the
# DLIO summary fallback — the fallback charges read-side startup + write-side
# cluster collection against the quiet window (this check's own docstring),
# so the number overstates the true gap on large topologies.
# ---------------------------------------------------------------------------


def test_a7_bookend_measured_breach_is_not_labeled_upper_bound(tmp_path):
    """A breach measured from real invocation bookends is the true quiet
    window — no upper-bound caveat belongs in the message."""
    write = (
        {"start": "2025-07-11T19:49:58", "end": "2025-07-11T19:50:12"},
        {"invocation_start_time": "2025-07-11T19:49:50",
         "invocation_end_time": "2025-07-11T19:50:00"},
        "20250711_194950",
    )
    read = (
        {"start": "2025-07-11T19:54:58", "end": "2025-07-11T19:55:12"},
        {"invocation_start_time": "2025-07-11T19:55:00",
         "invocation_end_time": "2025-07-11T19:55:10"},
        "20250711_195500",
    )
    check = _make_directory_check(tmp_path, [write, read])
    check.checkpointing_timestamp_gap_check()
    warned = check.log.warning.call_args[0][0]
    assert "upper bound" not in warned


def test_a7_summary_fallback_breach_is_labeled_upper_bound(tmp_path):
    """A breach measured via the DLIO summary fallback (metadata carries no
    invocation bookends — e.g. the submitter pruned metadata.json) must be
    labeled as an upper bound so reviewers know startup/collection overhead
    inflates the number (worklist A7 issue (b); the Microsoft AMLFS
    llama3-405b 128-node case)."""
    write = (
        {"start": "2025-07-11T19:49:58", "end": "2025-07-11T19:50:12"},
        {},
        "20250711_194950",
    )
    read = (
        {"start": "2025-07-11T19:54:58", "end": "2025-07-11T19:55:12"},
        {},
        "20250711_195500",
    )
    check = _make_directory_check(tmp_path, [write, read])
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True
    check.log.error.assert_not_called()
    assert check.log.warning.call_count == 1
    warned = check.log.warning.call_args[0][0]
    assert "upper bound" in warned, (
        "summary-fallback gap must be labeled an upper bound"
    )


def test_a7_unparseable_timestamps_remain_a_hard_error(tmp_path):
    """The A7 downgrade covers ONLY the gap breach; garbage timestamp data
    is still a hard 2.1.24 error (mirrors the #834 scope: missing/
    unparseable stay errors)."""
    checkpoint_files = [
        (
            {"start": "not-a-timestamp", "end": "2025-07-11T19:54:25"},
            {},
            "20250711_195047",
        ),
    ]
    check = _make_directory_check(tmp_path, checkpoint_files)
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is False
    check.log.error.assert_called_once()


def test_issue_812_falls_back_to_summary_when_bookends_absent(tmp_path):
    """Results dirs predating the invocation bookends (metadata without the
    keys) must still validate via the DLIO summary fallback — no crash, no
    parse-failure violation. Here the summary gap (2 min) is short relative to
    the ~3.5 min invocations, so it passes on the fallback path.
    """
    checkpoint_files = [
        (
            {"start": "2025-07-11T19:50:50", "end": "2025-07-11T19:54:25"},
            {},
            "20250711_195047",
        ),
        (
            {"start": "2025-07-11T19:56:25", "end": "2025-07-11T20:00:00"},
            {},
            "20250711_195625",
        ),
    ]
    check = _make_directory_check(tmp_path, checkpoint_files)
    valid = check.checkpointing_timestamp_gap_check()
    assert valid is True
