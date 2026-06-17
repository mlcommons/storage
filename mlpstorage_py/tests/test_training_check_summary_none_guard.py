"""Tests for BUG-T2: TrainingCheck methods must not crash when summary or metadata is None.

When summary.json or metadata.json is missing from a run timestamp,
``Loader.load_single_log`` returns None. The structural check
SubmissionStructureCheck reports the missing file under rule 2.1.19
(runFiles). Downstream TrainingCheck methods historically called
``summary.get(...)`` / ``metadata.get(...)`` without guarding, raising
``AttributeError: 'NoneType' object has no attribute 'get'`` six different
ways (one per method) — escaping run_checks() in tracebacks.

Fix: each method skips run-file tuples where the required dict is None,
relying on 2.1.19 to surface the underlying structural complaint.
"""

from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# ---------------------------------------------------------------------------
# Per BUG-T2 (validate-corpus traceback frames)
# ---------------------------------------------------------------------------

GUARDED_METHODS = [
    "recalculate_dataset_size",          # training_checks.py:135
    "run_data_matches_datasize",         # training_checks.py:255
    "accelerator_utilization_check",     # training_checks.py:296
    "single_host_simulated_accelerators",  # training_checks.py:325
    "single_host_client_limit",          # training_checks.py:371
    "identical_accelerators_per_node",   # training_checks.py:395
]


def _make_training_check(tmp_path, run_files):
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
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs)


@pytest.mark.parametrize("method_name", GUARDED_METHODS)
def test_bug_t2_method_does_not_crash_on_none_summary(method_name, tmp_path):
    """Each guarded method tolerates summary=None and returns truthy."""
    run_files = [(None, None, "20250101_130001")]
    check = _make_training_check(tmp_path, run_files)
    # Must NOT raise AttributeError.
    result = getattr(check, method_name)()
    # All guarded methods return True (valid) when there's nothing to check.
    assert result is True, f"{method_name} returned {result!r} for None-summary tuple"


@pytest.mark.parametrize("method_name", GUARDED_METHODS)
def test_bug_t2_method_does_not_crash_on_mixed_tuples(method_name, tmp_path):
    """A mix of None and well-formed tuples must not crash; well-formed ones still process."""
    run_files = [
        (None, None, "20250101_130001"),
        (
            {"num_accelerators": 1, "num_hosts": 1, "host_memory_GB": [64],
             "num_files_train": 0, "num_files_eval": 0,
             "metric": {"train_au_meet_expectation": "success",
                        "train_au_mean_percentage": 99.0}},
            {"combined_params": {
                "dataset": {"num_files_train": 1, "num_samples_per_file": 1,
                            "record_length_bytes": 1024},
                "reader": {"batch_size": 1}},
             "args": {"hosts": ["h1"]}},
            "20250101_140001",
        ),
    ]
    check = _make_training_check(tmp_path, run_files)
    # Must NOT raise.
    getattr(check, method_name)()
