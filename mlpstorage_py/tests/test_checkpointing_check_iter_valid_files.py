"""Tests for F1a: CheckpointingCheck._iter_valid_files sweep.

All 11 per-timestamp check methods in CheckpointingCheck previously did
``summary.get(...)`` / ``metadata.get(...)`` without guarding the case
where Loader returns None for a missing summary.json or metadata.json.
That produced one AttributeError traceback per affected submitter per
method — 13 methods × ~21 submitters = 273 tracebacks in the v2.0
reviewer corpus run, all escaping run_checks() into the log.

Fix: every loop iterates through ``self._iter_valid_files()`` which
silently skips None entries. The structural diagnostic for the missing
files is already owned by SubmissionStructureCheck (2.1.22 + 2.1.25).
"""

from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# All per-timestamp methods that were crashing in the v2.0 corpus run.
GUARDED_METHODS = [
    "checkpoint_data_size_ratio",       # 4.3.1
    "fsync_verification",               # 4.3.2
    "model_configuration_req",          # 4.3.3
    "closed_mpi_processes",             # 4.6.1
    "closed_accelerators_per_host",     # 4.6.2
    "aggregate_accelerator_memory",     # 4.3.4
    "closed_checkpoint_parameters",     # 4.6.3
    "checkpoint_path_args",             # 4.4.1
    "subset_run_validation",            # 4.3.5
    "open_mpi_processes",               # 4.6.4
    "checkpoint_filesystem_check",      # 4.4.2
]


def _make_checkpointing_check(tmp_path, checkpoint_files):
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
    return CheckpointingCheck(log=log, config=config, submissions_logs=submissions_logs)


def test_iter_valid_files_skips_none_entries(tmp_path):
    """Pure-generator behavior: None summary or metadata is filtered out."""
    files = [
        (None, None, "20250101_000000"),
        ({"x": 1}, None, "20250101_010000"),
        (None, {"y": 2}, "20250101_020000"),
        ({"x": 3}, {"y": 4}, "20250101_030000"),
    ]
    check = _make_checkpointing_check(tmp_path, files)
    yielded = list(check._iter_valid_files())
    assert yielded == [({"x": 3}, {"y": 4}, "20250101_030000")]


@pytest.mark.parametrize("method_name", GUARDED_METHODS)
def test_f1a_method_tolerates_all_none_entries(method_name, tmp_path):
    """Each guarded method completes without raising when every tuple is None."""
    checkpoint_files = [
        (None, None, "20250101_000000"),
        (None, None, "20250101_010000"),
    ]
    check = _make_checkpointing_check(tmp_path, checkpoint_files)
    result = getattr(check, method_name)()
    # All methods return True when there's nothing valid to check.
    assert result is True, (
        f"{method_name} returned {result!r} for all-None checkpoint_files; "
        "expected silent skip producing valid=True"
    )


@pytest.mark.parametrize("method_name", GUARDED_METHODS)
def test_f1a_method_tolerates_mixed_none_and_valid_entries(method_name, tmp_path):
    """Mixed None + well-formed tuples must not raise; well-formed entries process."""
    well_formed_summary = {
        "num_accelerators": 8, "num_hosts": 1, "host_memory_GB": [128],
        "metric": {"checkpoint_size_GB": 100},
        "start": "2025-01-01T00:00:00", "end": "2025-01-01T00:10:00",
    }
    well_formed_metadata = {
        "args": {"model": "llama3_8b", "num_processes": 8,
                 "checkpoint_folder": "/cf", "results_dir": "/rd"},
        "verification": "closed",
        "combined_params": {"checkpoint": {"fsync": True}},
        "params_dict": {"checkpoint.mode": "combined"},
        "yaml_params": {},
    }
    checkpoint_files = [
        (None, None, "20250101_000000"),
        (well_formed_summary, well_formed_metadata, "20250101_010000"),
    ]
    check = _make_checkpointing_check(tmp_path, checkpoint_files)
    # Must not raise.
    getattr(check, method_name)()
