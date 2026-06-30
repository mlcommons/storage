"""Issue #598: writer/reader contract on metadata.json parameter keys.

The writer at ``mlpstorage_py/benchmarks/base.py`` emits per-run
``*_metadata.json`` files with two top-level dicts:

    metadata["parameters"]           — YAML defaults with CLI overrides folded in
    metadata["override_parameters"]  — user-specified overrides only

(These key names were introduced as part of the #365 fix; the older names
``combined_params`` / ``params_dict`` are no longer written.)

The submission_checker rules under ``mlpstorage_py/submission_checker/checks/``
must read these same key names. Before #598 they still looked up the old
names, silently defaulted to ``{}``, and either fired spurious VIOLATIONs
("dataset parameters not found in metadata", "record length is 0",
"datagen size 0.00GiB is less than required 0.00GiB") or silently no-op'd
CLOSED/OPEN allow-list checks that should have fired.

This module locks both sides of the contract:

* ``TestWriterContract`` — ``BaseBenchmark.metadata`` emits the new key
  names and does NOT emit the old ones.
* ``TestReaderContract`` — the affected checker rules see populated values
  when the metadata uses the new keys, and do NOT spuriously violate.

Both halves were RED on the pre-fix tree (writer always passed; readers
all silently saw ``{}``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.checks.vdb_checks import VdbCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs

from mlpstorage_py.tests.conftest import MockLogger


# ---------------------------------------------------------------------------
# Writer-contract probe — drive a real Benchmark and inspect metadata dict
# ---------------------------------------------------------------------------


class _StubBenchmark(Benchmark):
    """Minimal concrete Benchmark for inspecting the metadata dict.

    Bypasses __init__ so we don't need a full args namespace; sets the
    handful of attributes the metadata property reads.
    """

    BENCHMARK_TYPE = None  # set on instance

    def _run(self):  # pragma: no cover — never executed
        pass


def _build_stub_benchmark(combined, overrides):
    """Construct a stub benchmark instance with combined_params and params_dict.

    Returns the instance with its metadata dict ready for inspection.
    """
    bench = _StubBenchmark.__new__(_StubBenchmark)
    bench.BENCHMARK_TYPE = SimpleNamespace(name="training")
    bench.args = SimpleNamespace(
        command="run", model="unet3d", num_processes=8, accelerator_type="h100"
    )
    bench.run_datetime = "20260630_120000"
    bench.run_result_output = "/tmp/results"
    bench.runtime = 0.0
    bench.verification = SimpleNamespace(name="closed")
    bench.executed_command = "mlpstorage training run ..."
    bench.command_output_files = []
    bench.cluster_information = None
    bench.cluster_snapshots = None
    bench.combined_params = combined
    bench.params_dict = overrides
    return bench


class TestWriterContract:
    """Writer side of the #598 contract."""

    def test_metadata_emits_new_key_names(self):
        """Real BaseBenchmark.metadata emits 'parameters' / 'override_parameters'."""
        bench = _build_stub_benchmark(
            combined={"dataset": {"num_files_train": 100}},
            overrides={"dataset.num_files_train": 200},
        )
        meta = bench.metadata

        assert "parameters" in meta, (
            "writer must emit metadata['parameters'] (renamed from "
            "'combined_params' as part of the #365 fix)"
        )
        assert "override_parameters" in meta, (
            "writer must emit metadata['override_parameters'] (renamed "
            "from 'params_dict' as part of the #365 fix)"
        )

    def test_metadata_does_not_emit_old_key_names(self):
        """Old names must NOT leak through — submitters relying on either
        name in tooling should get a hard miss, not a silent partial."""
        bench = _build_stub_benchmark(combined={}, overrides={})
        meta = bench.metadata

        assert "combined_params" not in meta, (
            "writer must not emit the pre-#365 'combined_params' key"
        )
        assert "params_dict" not in meta, (
            "writer must not emit the pre-#365 'params_dict' key"
        )

    def test_metadata_parameters_is_combined_plus_overrides(self):
        """parameters == combined_params with dotted overrides folded in.

        Pins the writer's documented semantics (fixes #365): downstream
        readers can treat 'parameters' as the single source of truth for
        the merged config.
        """
        bench = _build_stub_benchmark(
            combined={"dataset": {"num_files_train": 100}},
            overrides={"dataset.num_files_train": 999},
        )
        meta = bench.metadata
        assert meta["parameters"]["dataset"]["num_files_train"] == 999

    def test_metadata_override_parameters_is_user_overrides_only(self):
        bench = _build_stub_benchmark(
            combined={"dataset": {"num_files_train": 100}},
            overrides={"dataset.num_files_train": 999},
        )
        meta = bench.metadata
        assert meta["override_parameters"] == {"dataset.num_files_train": 999}

    def test_metadata_round_trips_through_json(self, tmp_path):
        """metadata is JSON-serialisable with the new key names."""
        bench = _build_stub_benchmark(
            combined={"dataset": {"num_files_train": 100}, "reader": {"batch_size": 4}},
            overrides={"dataset.num_files_train": 200},
        )
        path = tmp_path / "training_unet3d_metadata.json"
        path.write_text(json.dumps(bench.metadata), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["parameters"]["dataset"]["num_files_train"] == 200
        assert loaded["parameters"]["reader"]["batch_size"] == 4
        assert loaded["override_parameters"] == {"dataset.num_files_train": 200}


# ---------------------------------------------------------------------------
# Reader-contract probe — drive each affected checker against writer-shape input
# ---------------------------------------------------------------------------


def _writer_shape_training_metadata(
    *,
    parameters=None,
    override_parameters=None,
    verification="closed",
    hosts=("h1",),
    data_dir="/data",
    results_dir="/results",
):
    """Build a metadata dict shaped exactly like BaseBenchmark.metadata emits."""
    return {
        "benchmark_type": "training",
        "model": "unet3d",
        "verification": verification,
        "args": {
            "model": "unet3d",
            "num_processes": 8,
            "data_dir": data_dir,
            "results_dir": results_dir,
            "hosts": list(hosts),
        },
        "parameters": parameters if parameters is not None else {},
        "override_parameters": override_parameters if override_parameters is not None else {},
    }


def _writer_shape_checkpointing_metadata(
    *,
    parameters=None,
    override_parameters=None,
    verification="closed",
    model="llama3_8b",
    num_processes=8,
):
    return {
        "benchmark_type": "checkpointing",
        "model": model,
        "verification": verification,
        "args": {
            "model": model,
            "num_processes": num_processes,
            "checkpoint_folder": "/chkpts",
            "results_dir": "/results",
            "num_checkpoints_write": 1,
            "num_checkpoints_read": 0,
        },
        "parameters": parameters if parameters is not None else {},
        "override_parameters": override_parameters if override_parameters is not None else {},
    }


def _writer_shape_vdb_metadata(*, override_parameters=None, verification="closed"):
    return {
        "benchmark_type": "vector_database",
        "verification": verification,
        "args": {"storage_root": "/vdb/data", "results_dir": "/vdb/results"},
        "parameters": {},
        "override_parameters": override_parameters if override_parameters is not None else {},
    }


def _training_check(tmp_path, *, run_files=None, datagen_files=None, mode="training"):
    log = MagicMock()
    config = Config(version="v3.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=datagen_files or [],
        run_files=run_files or [],
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode=mode,
            benchmark="unet3d",
            folder=str(tmp_path),
        ),
    )
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _checkpointing_check(tmp_path, *, checkpoint_files=None, mode="checkpointing"):
    """CheckpointingCheck stores ``self.submissions_logs = submissions_logs.checkpoint_files``,
    so callers pass the checkpoint_files list directly (not run_files)."""
    log = MagicMock()
    config = Config(version="v3.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=[],
        checkpoint_files=checkpoint_files or [],
        system_file={},
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode=mode,
            benchmark="llama3_8b",
            folder=str(tmp_path),
        ),
    )
    return CheckpointingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _vdb_check(tmp_path, *, run_files=None, mock_logger=None):
    log = mock_logger or MockLogger()
    config = Config(
        version="v3.0", submitters=None, skip_output_file=True
    )
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files or [],
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="acme",
            system="sys-1",
            mode="vector_database",
            benchmark=tmp_path.name,
            folder=str(tmp_path),
        ),
    )
    return VdbCheck(log=log, config=config, submissions_logs=submissions_logs)


class TestReaderContract:
    """Reader side of the #598 contract — each affected rule must read the
    new key names emitted by the writer.

    Each test populates writer-shape data and asserts the corresponding
    rule does NOT fire a spurious violation. Before the fix, every test
    here was RED because the readers looked up the OLD names and got
    ``{}``, which made them either fire spuriously or silently pass an
    allow-list check.
    """

    # -- training_checks.py --------------------------------------------------

    def test_3_1_1_verify_datasize_usage_reads_parameters(self, tmp_path):
        """3.1.1 trainingVerifyDatasizeUsage — reads metadata['parameters']
        (was: 'combined_params' at lines 107 + 227)."""
        meta = _writer_shape_training_metadata(
            parameters={
                "dataset": {
                    "num_files_train": 100,
                    "num_samples_per_file": 1,
                    "record_length_bytes": 1024,
                }
            }
        )
        run_files = [({"num_accelerators": 1}, meta, "20260630_120000")]
        check = _training_check(tmp_path, run_files=run_files)

        result = check.verify_datasize_usage()

        assert result is True, (
            "3.1.1 must see populated dataset params under metadata['parameters'] "
            "and not fire 'dataset parameters not found in metadata'"
        )

    def test_3_1_2_recalculate_dataset_size_reads_parameters(self, tmp_path):
        """3.1.2 trainingRecalculateDatasetSize — reads metadata['parameters']
        (was: 'combined_params' at line 152)."""
        summary = {
            "num_accelerators": 1,
            "num_hosts": 1,
            "host_memory_GB": [64],
            "num_files_train": 1000,
            "num_files_eval": 0,
            "metric": {
                "train_au_meet_expectation": "success",
                "train_au_mean_percentage": 99.0,
            },
        }
        meta = _writer_shape_training_metadata(
            parameters={
                "dataset": {
                    "num_files_train": 1000,
                    "num_samples_per_file": 1,
                    "record_length_bytes": 1024 * 1024,
                },
                "reader": {"batch_size": 4},
            }
        )
        check = _training_check(
            tmp_path, run_files=[(summary, meta, "20260630_120000")]
        )

        check.recalculate_dataset_size()

        # The pre-fix failure was 'record length is 0' — a spurious violation
        # because record_length_bytes was read from empty combined_params.
        all_error_calls = [str(c) for c in check.log.error.call_args_list]
        assert not any("record length is 0" in m for m in all_error_calls), (
            "3.1.2 must read record_length_bytes from metadata['parameters'] "
            f"and not fire 'record length is 0'; got: {all_error_calls}"
        )

    def test_3_2_1_datagen_minimum_size_reads_parameters(self, tmp_path):
        """3.2.1 trainingDatagenMinimumSize — reads metadata['parameters']
        (was: 'combined_params' at lines 227 + 238)."""
        params = {
            "dataset": {
                "num_files_train": 1000,
                "num_samples_per_file": 1,
                "record_length_bytes": 1024 * 1024,
            }
        }
        run_meta = _writer_shape_training_metadata(parameters=params)
        datagen_meta = _writer_shape_training_metadata(parameters=params)
        check = _training_check(
            tmp_path,
            run_files=[({}, run_meta, "20260630_120000")],
            datagen_files=[({}, datagen_meta, "20260630_110000")],
        )

        result = check.datagen_minimum_size()

        # Pre-fix: both run-side and datagen-side dataset_params read 0, so
        # expected_size == datagen_size == 0.0, and the spurious-zero compare
        # at line 244 was `0 < 0` (False) — silent pass. But more important
        # is that mis-equal sizing would fire. Assert no datagen_size
        # violation fired regardless.
        assert result is True

    def test_3_6_2_closed_submission_parameters_reads_override_parameters(self, tmp_path):
        """3.6.2 trainingClosedSubmissionParameters — reads metadata['override_parameters']
        (was: 'params_dict' at line 579).

        Populates an override_parameters dict with a DISALLOWED key. Pre-fix
        the reader saw ``{}`` and the allow-list check silently passed — a
        false negative that let CLOSED submissions ship illegal overrides.
        """
        # 'dataset.format' is in OPEN-extra allowed, not CLOSED-allowed.
        meta = _writer_shape_training_metadata(
            override_parameters={"dataset.format": "tfrecord"},
            verification="closed",
        )
        check = _training_check(
            tmp_path, run_files=[({}, meta, "20260630_120000")]
        )

        result = check.closed_submission_parameters()

        assert result is False, (
            "3.6.2 must read disallowed override from metadata['override_parameters'] "
            "and fire a violation (pre-fix this silently passed)"
        )
        violations = [
            str(c) for c in check.log.error.call_args_list
            if "dataset.format" in str(c)
        ]
        assert violations, (
            "Expected a [3.6.2 trainingClosedSubmissionParameters] violation "
            f"mentioning the disallowed override; got: {check.log.error.call_args_list}"
        )

    def test_3_6_3_open_submission_parameters_reads_override_parameters(self, tmp_path):
        """3.6.3 trainingOpenSubmissionParameters — reads metadata['override_parameters']
        (was: 'params_dict' at line 642)."""
        # 'reader.something_made_up' is not in OPEN allowed list.
        meta = _writer_shape_training_metadata(
            override_parameters={"reader.something_made_up": "value"},
            verification="open",
        )
        check = _training_check(
            tmp_path, run_files=[({}, meta, "20260630_120000")]
        )

        result = check.open_submission_parameters()

        assert result is False, (
            "3.6.3 must read disallowed override from metadata['override_parameters'] "
            "and fire a violation"
        )

    # -- checkpointing_checks.py --------------------------------------------

    def test_4_3_2_fsync_verification_reads_parameters(self, tmp_path):
        """4.3.2 checkpointFsyncVerification — reads metadata['parameters']
        (was: 'combined_params' at line 160).

        Pre-fix the reader silently saw ``{}`` and fsync_enabled defaulted
        to False, firing a spurious violation on every checkpoint run.
        """
        meta = _writer_shape_checkpointing_metadata(
            parameters={"checkpoint": {"fsync": True}}
        )
        summary = {
            "num_accelerators": 8,
            "num_hosts": 1,
            "host_memory_GB": [128],
            "metric": {"checkpoint_size_GB": 100},
            "start": "2026-06-30T00:00:00",
            "end": "2026-06-30T00:10:00",
        }
        check = _checkpointing_check(
            tmp_path, checkpoint_files=[(summary, meta, "20260630_120000")]
        )

        result = check.fsync_verification()

        assert result is True, (
            "4.3.2 must read fsync=True from metadata['parameters']['checkpoint'] "
            "and not fire a spurious 'fsync not enabled' violation"
        )

    def test_4_3_5_subset_run_validation_reads_override_parameters(self, tmp_path):
        """4.3.5 checkpointSubsetRunValidation — reads metadata['override_parameters']
        (was: 'params_dict' at line 504)."""
        meta = _writer_shape_checkpointing_metadata(
            override_parameters={"checkpoint.mode": "subset"},
            model="llama3_70b",
        )
        # Subset mode requires exactly 8 accelerators — supply 4 to trigger
        # the violation. If the reader silently saw {} (pre-fix), it would
        # never enter the if-branch at all and silent-pass.
        summary = {"num_accelerators": 4}
        check = _checkpointing_check(
            tmp_path, checkpoint_files=[(summary, meta, "20260630_120000")]
        )

        result = check.subset_run_validation()

        assert result is False, (
            "4.3.5 must see checkpoint.mode='subset' in metadata['override_parameters'] "
            "and fire on num_accelerators != 8"
        )

    def test_4_6_1_closed_mpi_processes_reads_override_parameters(self, tmp_path):
        """4.6.1 checkpointClosedMpiProcesses — reads metadata['override_parameters']
        (was: 'params_dict' at line 223)."""
        # subset mode + wrong num_processes → expect violation.
        meta = _writer_shape_checkpointing_metadata(
            override_parameters={"checkpoint.mode": "subset"},
            model="llama3_70b",
            num_processes=4,
        )
        summary = {"num_accelerators": 4}
        check = _checkpointing_check(
            tmp_path, checkpoint_files=[(summary, meta, "20260630_120000")]
        )

        result = check.closed_mpi_processes()

        assert result is False, (
            "4.6.1 must read checkpoint.mode from metadata['override_parameters'] "
            "and require num_processes=8 for subset mode"
        )

    # -- vdb_checks.py -------------------------------------------------------

    def test_5_6_4_vdb_closed_submission_parameters_reads_override_parameters(self, tmp_path):
        """5.6.4 vdbClosedSubmissionParameters — reads metadata['override_parameters']
        (was: 'params_dict' at line 880)."""
        # Pick a key very unlikely to be in _CLOSED_ALLOWED_PARAMS.
        meta = _writer_shape_vdb_metadata(
            override_parameters={"definitely.not.allowed.key": "x"},
        )
        summary = {"database": {"database": "milvus"}}
        check = _vdb_check(
            tmp_path, run_files=[(summary, meta, "20260630_120000")]
        )

        result = check.vdb_closed_submission_parameters()

        assert result is False, (
            "5.6.4 must read disallowed override from metadata['override_parameters'] "
            "and fire a violation"
        )

    def test_5_6_5_vdb_open_submission_parameters_reads_override_parameters(self, tmp_path):
        """5.6.5 vdbOpenSubmissionParameters — reads metadata['override_parameters']
        (was: 'params_dict' at line 941)."""
        meta = _writer_shape_vdb_metadata(
            override_parameters={"definitely.not.allowed.key": "x"},
            verification="open",
        )
        summary = {"database": {"database": "milvus"}}
        # Override the division to "open" for this check.
        log = MagicMock()
        config = Config(version="v3.0", submitters=None, skip_output_file=True)
        submissions_logs = SubmissionLogs(
            datagen_files=[],
            run_files=[(summary, meta, "20260630_120000")],
            system_file=None,
            loader_metadata=LoaderMetadata(
                division="open",
                submitter="acme",
                system="sys-1",
                mode="vector_database",
                benchmark=tmp_path.name,
                folder=str(tmp_path),
            ),
        )
        check = VdbCheck(log=log, config=config, submissions_logs=submissions_logs)

        result = check.vdb_open_submission_parameters()

        assert result is False, (
            "5.6.5 must read disallowed override from metadata['override_parameters'] "
            "and fire a violation"
        )
