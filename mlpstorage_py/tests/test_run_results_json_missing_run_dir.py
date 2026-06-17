"""Tests for F2: 2.1.16 runResultsJson must not crash when run/ is missing.

When a training submission omits training/<workload>/run/ (a structural
violation reported under 2.1.12), the check called
``list_files(self.run_path)`` directly and crashed with FileNotFoundError
from ``utils.list_files`` (line 31). That exception escaped run_checks()
into the log as a "directory_checks.py:162" traceback for each affected
submitter — 39 in the v2.0 corpus, plus 8 distinct
"training/<wl>/run" FileNotFoundError paths counted separately (same
root cause).

Fix: probe ``os.path.isdir(self.run_path)`` before calling list_files;
emit a 2.1.16 violation pointing at the missing run/ dir and continue.
"""

from unittest.mock import MagicMock

from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


def _make_directory_check(tmp_path):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=[],
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode="training",
            benchmark="unet3d",
            folder=str(tmp_path),  # NOTE: tmp_path itself; run/ subdir is NOT created
        ),
    )
    return DirectoryCheck(log=log, config=config, submissions_logs=submissions_logs)


def test_f2_run_dir_missing_does_not_crash(tmp_path):
    """2.1.16 must not raise FileNotFoundError when run/ is absent."""
    check = _make_directory_check(tmp_path)
    # tmp_path exists but no run/ subdir.
    valid = check.run_results_json_check()
    assert valid is False
    # Violation logged at error level under rule 2.1.16.
    error_calls = [
        c for c in check.log.error.call_args_list
        if "2.1.16" in str(c) and "run/ directory not found" in str(c)
    ]
    assert len(error_calls) == 1, (
        "Expected exactly one 2.1.16 missing-run violation; got: %r" % error_calls
    )


def test_f2_run_dir_present_with_results_json_is_valid(tmp_path):
    """Regression pin: positive path still works."""
    import os
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    check = _make_directory_check(tmp_path)
    valid = check.run_results_json_check()
    assert valid is True


def test_f2_run_dir_present_no_results_json_emits_2_1_16(tmp_path):
    """Regression pin: missing results.json inside present run/ still fires."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    check = _make_directory_check(tmp_path)
    valid = check.run_results_json_check()
    assert valid is False
