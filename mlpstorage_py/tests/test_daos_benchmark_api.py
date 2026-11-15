"""
Tests for systems that declare ``benchmark_API: daos`` (native DAOS access
through libdfs).

Covers:
  - 3.4.2 trainingMlpstorageFilesystemCheck needs no filesystem-separation
    evidence when every run's metadata records a DAOS storage config
    (storage_type 'daos', non-empty pool and container), and records a
    violation for a run that does not.
  - 4.4.2 checkpointFilesystemCheck behaves the same way.

Run with:
    pytest mlpstorage_py/tests/test_daos_benchmark_api.py -v
"""

import json

from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
from mlpstorage_py.submission_checker.checks.helpers import daos_storage_mismatch
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader

_DAOS_STORAGE = {
    "storage_type": "daos",
    "storage_root": "/",
    "storage_options": {"pool": "pool0", "container": "mlperf"},
}


def _load_check(root_path, mock_logger, mode, check_cls):
    config = Config(submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path))
    for logs in loader.load():
        if logs.loader_metadata.mode == mode:
            return check_cls(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError(f"no {mode} SubmissionLogs yielded from the fixture")


def _set_storage(root, pattern, storage, only_first=False):
    paths = sorted(root.glob(pattern))
    assert paths, f"fixture has no metadata.json matching {pattern}"
    for path in paths[:1] if only_first else paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("parameters", {})["storage"] = storage
        path.write_text(json.dumps(meta), encoding="utf-8")


_TRAINING_RUNS = "closed/Acme/results/*/training/*/run/*/metadata.json"
_CHECKPOINT_RUNS = "closed/Acme/results/*/checkpointing/*/*/metadata.json"


def _checkpoint_submission(tmp_path):
    from mlpstorage_py.tests.conftest import build_submission
    return build_submission(
        tmp_path,
        benchmark_api="daos",
        chkpt_checkpoint_folder="/chkpts",
        chkpt_results_dir="/results",
    )


def test_training_daos_api_passes_with_daos_metadata(tmp_path, mock_logger):
    """benchmark_API='daos' + DAOS storage in every run -> 3.4.2 passes."""
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path, benchmark_api="daos")
    _set_storage(root, _TRAINING_RUNS, _DAOS_STORAGE)
    check = _load_check(root, mock_logger, "training", TrainingCheck)
    assert check.mlpstorage_filesystem_check() is True
    assert mock_logger.errors == []


def test_training_daos_api_flags_non_daos_run(tmp_path, mock_logger):
    """benchmark_API='daos' but a run used another backend -> 3.4.2 violation."""
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path, benchmark_api="daos")
    _set_storage(root, _TRAINING_RUNS, _DAOS_STORAGE)
    _set_storage(root, _TRAINING_RUNS, {"storage_type": "local_fs"}, only_first=True)
    check = _load_check(root, mock_logger, "training", TrainingCheck)
    assert check.mlpstorage_filesystem_check() is False
    assert any("3.4.2" in e and "local_fs" in e for e in mock_logger.errors)


def test_training_daos_api_flags_missing_storage(tmp_path, mock_logger):
    """benchmark_API='daos' with no recorded storage config -> 3.4.2 violation."""
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path, benchmark_api="daos")
    check = _load_check(root, mock_logger, "training", TrainingCheck)
    assert check.mlpstorage_filesystem_check() is False
    assert any("3.4.2" in e for e in mock_logger.errors)


def test_checkpointing_daos_api_passes_with_daos_metadata(tmp_path, mock_logger):
    """benchmark_API='daos' + DAOS storage in every run -> 4.4.2 passes."""
    root = _checkpoint_submission(tmp_path)
    _set_storage(root, _CHECKPOINT_RUNS, _DAOS_STORAGE)
    check = _load_check(root, mock_logger, "checkpointing", CheckpointingCheck)
    assert check.checkpoint_filesystem_check() is True
    assert mock_logger.errors == []


def test_checkpointing_daos_api_flags_empty_container(tmp_path, mock_logger):
    """benchmark_API='daos' but a run recorded no container -> 4.4.2 violation."""
    root = _checkpoint_submission(tmp_path)
    _set_storage(root, _CHECKPOINT_RUNS, _DAOS_STORAGE)
    _set_storage(
        root, _CHECKPOINT_RUNS,
        {"storage_type": "daos", "storage_options": {"pool": "pool0", "container": ""}},
        only_first=True,
    )
    check = _load_check(root, mock_logger, "checkpointing", CheckpointingCheck)
    assert check.checkpoint_filesystem_check() is False
    assert any("4.4.2" in e and "container" in e for e in mock_logger.errors)


def test_daos_storage_mismatch_reasons():
    assert daos_storage_mismatch({"parameters": {"storage": _DAOS_STORAGE}}) is None
    assert "expected 'daos'" in daos_storage_mismatch({})
    assert "expected 'daos'" in daos_storage_mismatch(
        {"parameters": {"storage": {"storage_type": "s3"}}}
    )
    reason = daos_storage_mismatch(
        {"parameters": {"storage": {"storage_type": "daos", "storage_options": {"pool": "p"}}}}
    )
    assert "storage.storage_options.container" in reason
    assert "pool" not in reason.replace("storage_options", "")


def test_file_api_still_requires_evidence(tmp_path, mock_logger):
    """benchmark_API='file' without evidence stays INVALID (unchanged)."""
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path, run_data_dir="/data", run_results_dir="/results")
    check = _load_check(root, mock_logger, "training", TrainingCheck)
    assert check.mlpstorage_filesystem_check() is False
