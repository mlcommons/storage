"""Wiring: the datagen manifest reaches the DLIO command line on ``training run``.

Instantiates a real ``TrainingBenchmark`` (cluster info mocked, base CAP gate
patched to a no-op so only the training override's manifest hook runs) and
drives ``_pre_execution_gate()`` — the point in ``Benchmark.run()`` right after
CAP-02 — to prove the injected ``dataset.num_files_generated`` lands in
``generate_dlio_command()`` output.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.rules.datagen_hierarchy import DATAGEN_MANIFEST_FILENAME, write_datagen_manifest
from tests.fixtures.sample_data import create_sample_benchmark_args


@pytest.fixture(autouse=True)
def _bypass_base_gate():
    with patch("mlpstorage_py.benchmarks.base.Benchmark._pre_execution_gate", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _mock_cluster_information():
    with patch("mlpstorage_py.benchmarks.base.ClusterInformation") as mock_ci:
        mock_ci.return_value = MagicMock()
        mock_ci.return_value.total_memory_bytes = 256 * 1024**3
        mock_ci.return_value.host_info_list = []
        yield mock_ci


def _run_benchmark(tmp_path, *, command="run", mode="closed"):
    args = create_sample_benchmark_args(
        benchmark_type="training", command=command, model="unet3d",
        accelerator_type="b200", num_accelerators=8, client_host_memory_in_gb=256,
        hosts=["127.0.0.1"], data_dir=str(tmp_path / "data"),
    )
    args.mode = mode
    args.results_dir = str(tmp_path / "results")
    args.dry_run = True
    args.what_if = mode == "whatif"
    (tmp_path / "data").mkdir(exist_ok=True)
    from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
    logger = MagicMock()
    return TrainingBenchmark(args, logger=logger), logger


def _write_manifest(benchmark, generated, **over):
    ds = dict(benchmark.combined_params["dataset"])
    ds["num_files_train"] = generated
    ds.update(over)
    return write_datagen_manifest(benchmark.args.data_dir, "unet3d", ds, "/results/leaf")


class TestRunWiring:
    def test_manifest_count_reaches_the_dlio_command(self, tmp_path):
        benchmark, logger = _run_benchmark(tmp_path)
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count + 5)

        benchmark._pre_execution_gate()

        assert benchmark.params_dict["dataset.num_files_generated"] == run_count + 5
        assert benchmark.params_dict["dataset.num_files_train"] == run_count
        cmd = benchmark.generate_dlio_command()
        assert f"++workload.dataset.num_files_generated={run_count + 5}" in cmd
        assert f"++workload.dataset.num_files_train={run_count}" in cmd

    def test_missing_manifest_warns_and_leaves_the_command_alone(self, tmp_path):
        benchmark, logger = _run_benchmark(tmp_path)

        benchmark._pre_execution_gate()

        warnings = " ".join(str(c.args[0]) for c in logger.warning.call_args_list)
        assert "MANIFEST-000" in warnings
        assert os.path.join(benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME) in warnings
        assert "num_files_generated" not in benchmark.generate_dlio_command()

    def test_shortfall_aborts_before_dlio_launches(self, tmp_path):
        benchmark, _ = _run_benchmark(tmp_path)
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count - 1)

        with pytest.raises(ConfigurationError) as ei:
            benchmark._pre_execution_gate()
        text = str(ei.value)
        assert "MANIFEST-003" in text
        # The re-datagen hint is a real, parseable mlpstorage command.
        assert "mlpstorage closed training unet3d datagen file" in text
        assert f"dataset.num_files_train={run_count}" in text
        assert "num_files_generated" not in benchmark.generate_dlio_command()

    def test_whatif_shortfall_is_a_warning(self, tmp_path):
        benchmark, logger = _run_benchmark(tmp_path, mode="whatif")
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count - 1)

        benchmark._pre_execution_gate()

        assert "MANIFEST-003" in " ".join(str(c.args[0]) for c in logger.warning.call_args_list)

    def test_datagen_gate_never_reads_a_manifest(self, tmp_path):
        benchmark, _ = _run_benchmark(tmp_path, command="datagen")
        with patch("mlpstorage_py.benchmarks.dlio.read_datagen_manifest") as read:
            benchmark._pre_execution_gate()
            read.assert_not_called()

    def test_run_metadata_records_the_injection(self, tmp_path):
        benchmark, _ = _run_benchmark(tmp_path)
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count + 1)
        benchmark._pre_execution_gate()

        benchmark.write_metadata()
        ts = os.path.basename(benchmark.run_result_output)
        meta = json.load(open(os.path.join(benchmark.run_result_output, f"training_{ts}_metadata.json")))
        assert meta["override_parameters"]["dataset.num_files_generated"] == run_count + 1
        assert meta["parameters"]["dataset"]["num_files_generated"] == run_count + 1


class TestSnapshotWiring:
    """D4 linkage: a real run leaf gets ``datagen-manifest.json`` and its
    metadata declares it; configview writes nothing into a leaf."""

    def test_run_leaf_carries_the_consumed_manifest(self, tmp_path):
        from mlpstorage_py.rules.datagen_hierarchy import (
            DATAGEN_MANIFEST_SNAPSHOT_FILENAME, METADATA_DATAGEN_MANIFEST_KEY,
            read_datagen_manifest_snapshot,
        )
        benchmark, _ = _run_benchmark(tmp_path)
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count + 5)
        benchmark._pre_execution_gate()
        benchmark.write_metadata()
        leaf = benchmark.run_result_output
        snapshot = read_datagen_manifest_snapshot(leaf)
        assert snapshot is not None
        assert snapshot.manifest.num_files_train == run_count + 5
        assert snapshot.manifest.location == os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME)
        assert snapshot.overridden == []
        ts = os.path.basename(leaf)
        meta = json.load(open(os.path.join(leaf, f"training_{ts}_metadata.json")))
        assert meta[METADATA_DATAGEN_MANIFEST_KEY] == DATAGEN_MANIFEST_SNAPSHOT_FILENAME
        assert meta["parameters"]["dataset"]["num_files_generated"] == run_count + 5

    def test_configview_writes_no_snapshot(self, tmp_path):
        from mlpstorage_py.rules.datagen_hierarchy import DATAGEN_MANIFEST_SNAPSHOT_FILENAME
        benchmark, _ = _run_benchmark(tmp_path, command="configview")
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count + 5)
        benchmark._pre_execution_gate()
        assert benchmark.params_dict["dataset.num_files_generated"] == run_count + 5
        assert not os.path.exists(os.path.join(benchmark.run_result_output,
                                               DATAGEN_MANIFEST_SNAPSHOT_FILENAME))
        assert "datagen_manifest_file" not in benchmark.metadata

    def test_whatif_shortfall_snapshot_records_the_override(self, tmp_path):
        from mlpstorage_py.rules.datagen_hierarchy import read_datagen_manifest_snapshot
        benchmark, _ = _run_benchmark(tmp_path, mode="whatif")
        run_count = int(benchmark.combined_params["dataset"]["num_files_train"])
        _write_manifest(benchmark, run_count - 1)
        benchmark._pre_execution_gate()
        snapshot = read_datagen_manifest_snapshot(benchmark.run_result_output)
        assert snapshot.overridden == ["MANIFEST-003"]
