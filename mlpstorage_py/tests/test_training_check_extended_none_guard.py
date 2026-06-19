"""Tests for F1b: extend TrainingCheck None-guard to remaining 7 loop sites.

BUG-T2 covered the 6 sites surfaced by the first validate run. The
corpus-wide re-run unmasked 5 more methods (training_checks.py:88, 512,
563, 593, 642) plus datagen_minimum_size's two internal loops (run_files
and datagen_files) — all crashing with the same
``AttributeError: 'NoneType' object has no attribute 'get'`` pattern when
summary.json / metadata.json is missing.

Fix: inline ``if metadata is None: continue`` (or no-debug skip for the
internal datagen_minimum_size loops) at each remaining loop top. Matches
BUG-T2's pattern.
"""

from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# Methods unmasked by the corpus-wide validate re-run.
F1B_GUARDED_METHODS = [
    "verify_datasize_usage",           # 3.1.1   line 86
    "datagen_minimum_size",            # 3.2.1   lines 206/215 (two internal loops)
    "closed_submission_parameters",    # 3.6.2   line 512
    "open_submission_parameters",      # 3.6.3   line 563
    "mlpstorage_path_args",            # 3.4.1   line 593
    "mlpstorage_filesystem_check",     # 3.4.2   line 642
]


def _make_training_check(tmp_path, run_files, datagen_files=None):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=datagen_files or [],
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


@pytest.mark.parametrize("method_name", F1B_GUARDED_METHODS)
def test_f1b_method_tolerates_all_none_run_files(method_name, tmp_path):
    """Each method completes without raising when every run/datagen tuple is None."""
    none_files = [(None, None, "20250101_000000"), (None, None, "20250101_010000")]
    check = _make_training_check(tmp_path, none_files, datagen_files=none_files)
    # Must not raise.
    getattr(check, method_name)()


@pytest.mark.parametrize("method_name", F1B_GUARDED_METHODS)
def test_f1b_method_tolerates_mixed_none_and_well_formed(method_name, tmp_path):
    """Mixed None + well-formed tuples must not raise."""
    well_formed_summary = {
        "num_accelerators": 1, "num_hosts": 1, "host_memory_GB": [64],
        "num_files_train": 0, "num_files_eval": 0,
        "metric": {"train_au_meet_expectation": "success",
                   "train_au_mean_percentage": 99.0},
    }
    well_formed_metadata = {
        "combined_params": {
            "dataset": {"num_files_train": 1, "num_samples_per_file": 1,
                        "record_length_bytes": 1024},
            "reader": {"batch_size": 1},
        },
        "args": {"hosts": ["h1"], "data_dir": "/data", "results_dir": "/results"},
        "verification": "open",
        "params_dict": {},
    }
    run_files = [
        (None, None, "20250101_000000"),
        (well_formed_summary, well_formed_metadata, "20250101_010000"),
    ]
    datagen_files = [
        (None, None, "20250101_120000"),
        (well_formed_summary, well_formed_metadata, "20250101_130000"),
    ]
    check = _make_training_check(tmp_path, run_files, datagen_files=datagen_files)
    # Must not raise.
    getattr(check, method_name)()
