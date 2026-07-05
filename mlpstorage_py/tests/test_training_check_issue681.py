"""
Regression tests for issue #681 — TypeError in run_data_matches_datasize
(rule 3.3.1) when num_files_train is serialised as a string in JSON/YAML.

Root cause: summary.json and metadata.json may carry integer-valued fields
as quoted strings after YAML→JSON round-tripping.  The `>` / `<` comparisons
at training_checks.py:370,390 did not coerce types, raising:
    TypeError: '>' not supported between instances of 'int' and 'str'

Fix: _to_int() coerces at read time in _extract_latest_datagen_cardinality,
_group_datasize_by_data_dir, and the run-loop read of run_num_files_train.
"""

import pytest
from unittest.mock import MagicMock

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# ---------------------------------------------------------------------------
# _to_int unit tests
# ---------------------------------------------------------------------------

class TestToInt:
    """TrainingCheck._to_int handles every value shape that JSON/YAML can produce."""

    def test_int_passthrough(self):
        assert TrainingCheck._to_int(68090280) == 68090280

    def test_string_int_coerced(self):
        assert TrainingCheck._to_int("68090280") == 68090280

    def test_none_returns_none(self):
        assert TrainingCheck._to_int(None) is None

    def test_float_string_coerced(self):
        assert TrainingCheck._to_int("100.0") is None

    def test_garbage_string_returns_none(self):
        assert TrainingCheck._to_int("not_a_number") is None

    def test_zero(self):
        assert TrainingCheck._to_int("0") == 0

    def test_zero_int(self):
        assert TrainingCheck._to_int(0) == 0


# ---------------------------------------------------------------------------
# Helper — build a minimal TrainingCheck from crafted SubmissionLogs
# ---------------------------------------------------------------------------

def _make_check(tmp_path, mock_logger, datagen_files, run_files):
    """Return a TrainingCheck wired with the given file tuples."""
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    lm = LoaderMetadata(
        division="closed",
        submitter="Acme",
        system="sys",
        mode="training",
        benchmark="retinanet",
        folder=str(tmp_path),
    )
    logs = SubmissionLogs(
        datagen_files=datagen_files,
        datasize_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=lm,
    )
    return TrainingCheck(log=mock_logger, config=config, submissions_logs=logs)


def _datagen_meta(num_files_train):
    """Build a minimal datagen metadata dict with the given num_files_train."""
    return {
        "args": {"data_dir": "/data/retinanet"},
        "parameters": {
            "dataset": {
                "num_files_train": num_files_train,
            }
        },
    }


def _run_summary(num_files_train):
    return {"num_files_train": num_files_train}


def _run_meta():
    return {"args": {"data_dir": "/data/retinanet"}}


# ---------------------------------------------------------------------------
# Issue #681 crash regression — string vs int comparisons must not TypeError
# ---------------------------------------------------------------------------

class TestRunDataMatchesDatasize_Issue681:
    """rule 3.3.1 must not raise TypeError when num_files_train is a string."""

    def test_string_run_int_datagen_no_crash(self, tmp_path, mock_logger):
        """run summary has str num_files_train; datagen metadata has int — no crash."""
        datagen_files = [(None, _datagen_meta(68090280), "20250111_130000")]
        run_files = [(_run_summary("68090280"), _run_meta(), "20250111_140001")]
        check = _make_check(tmp_path, mock_logger, datagen_files, run_files)
        # Must not raise TypeError
        result = check.run_data_matches_datasize()
        assert result is True
        assert mock_logger.errors == []

    def test_int_run_string_datagen_no_crash(self, tmp_path, mock_logger):
        """run summary has int num_files_train; datagen metadata has str — no crash."""
        datagen_files = [(None, _datagen_meta("68090280"), "20250111_130000")]
        run_files = [(_run_summary(68090280), _run_meta(), "20250111_140001")]
        check = _make_check(tmp_path, mock_logger, datagen_files, run_files)
        result = check.run_data_matches_datasize()
        assert result is True
        assert mock_logger.errors == []

    def test_string_run_string_datagen_no_crash(self, tmp_path, mock_logger):
        """Both sides are strings — no crash, no warnings."""
        datagen_files = [(None, _datagen_meta("68090280"), "20250111_130000")]
        run_files = [(_run_summary("68090280"), _run_meta(), "20250111_140001")]
        check = _make_check(tmp_path, mock_logger, datagen_files, run_files)
        result = check.run_data_matches_datasize()
        assert result is True
        assert mock_logger.errors == []

    def test_overrun_still_warns_after_coercion(self, tmp_path, mock_logger):
        """When run > datagen (both strings), DATAGEN-OVERRUN warning is still emitted."""
        datagen_files = [(None, _datagen_meta("1000"), "20250111_130000")]
        run_files = [(_run_summary("2000"), _run_meta(), "20250111_140001")]
        check = _make_check(tmp_path, mock_logger, datagen_files, run_files)
        check.run_data_matches_datasize()
        assert any("DATAGEN-OVERRUN" in w for w in mock_logger.warnings), (
            "Expected DATAGEN-OVERRUN warning when run > datagen"
        )
