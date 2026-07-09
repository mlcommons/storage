"""Integration tests: datagen_hierarchy guards wired into TrainingBenchmark.

The unit tests in ``tests/unit/test_datagen_hierarchy.py`` prove the four
helpers in ``mlpstorage_py.rules.datagen_hierarchy`` behave correctly in
isolation. This file proves that the training benchmark's
``__init__`` and ``_run`` actually call them at the right points and only
for the right command/mode combinations:

    - Pre-run guards fire for ``command == "datagen"``.
    - Post-run manifest write + leaf WARN fire after datagen succeeds
      (before ``_run`` returns SUCCESS).
    - Whatif mode skips all guards (matches D-29 reportgen policy and
      the user's "no significant validation checking" direction).
    - Object storage skips the local manifest write (S3 presence
      semantics are a different API).
    - Non-datagen commands (``run``, ``configview``, ``datasize``) do
      not invoke datagen-specific guards — a populated ``--data-dir``
      is expected input for ``run``, not a refuse-to-overwrite signal.
"""
from __future__ import annotations

import json
import os

import pytest
from unittest.mock import MagicMock, patch

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.rules.datagen_hierarchy import DATAGEN_MANIFEST_FILENAME
from tests.fixtures.sample_data import create_sample_benchmark_args


@pytest.fixture(autouse=True)
def _bypass_capacity_gate():
    """Skip the pre-execution CAP gate for these instantiation tests."""
    with patch(
        "mlpstorage_py.benchmarks.base.Benchmark._pre_execution_gate",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _mock_cluster_information():
    """Skip the real ClusterInformation collection."""
    with patch("mlpstorage_py.benchmarks.base.ClusterInformation") as mock_ci:
        mock_ci.return_value = MagicMock()
        mock_ci.return_value.total_memory_bytes = 256 * 1024**3
        mock_ci.return_value.host_info_list = []
        yield mock_ci


def _training_datagen_args(tmp_path, model="unet3d", mode="closed"):
    """Build a training-datagen args Namespace with results/data dirs under tmp_path."""
    args = create_sample_benchmark_args(
        benchmark_type="training",
        command="datagen",
        model=model,
        accelerator_type="h100",
        num_accelerators=8,
        client_host_memory_in_gb=256,
        hosts=["127.0.0.1"],
        data_dir=str(tmp_path / "data"),
    )
    args.mode = mode
    args.results_dir = str(tmp_path / "results")
    args.dry_run = True
    args.what_if = mode == "whatif"
    return args


# --------------------------------------------------------------------------- #
# Pre-run guard — refuse-to-overwrite an existing <data-dir>/<model>          #
# --------------------------------------------------------------------------- #


class TestRefuseToOverwriteWiring:
    """<data-dir>/<model> already exists → ``TrainingBenchmark.__init__`` raises."""

    def test_datagen_closed_refuses_populated_data_dir(self, tmp_path):
        # Pre-populate <data-dir>/unet3d/ with a stray file to simulate a
        # prior datagen (or a hand-crafted tree).
        (tmp_path / "data" / "unet3d").mkdir(parents=True)
        (tmp_path / "data" / "unet3d" / "prior_shard.npz").write_bytes(b"x")

        args = _training_datagen_args(tmp_path)
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with pytest.raises(ConfigurationError) as excinfo:
            TrainingBenchmark(args, logger=MagicMock())

        text = str(excinfo.value)
        # The guard's language, not some downstream reason.
        assert "unet3d" in text
        assert (
            "refusing to overwrite" in text
            or "already exists" in text
        ), f"unexpected error text: {text!r}"

    def test_datagen_closed_accepts_empty_data_dir(self, tmp_path):
        # <data-dir> exists but <data-dir>/unet3d does not — the fresh case.
        (tmp_path / "data").mkdir(parents=True)

        args = _training_datagen_args(tmp_path)
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        # Patch the guard so we can prove it was called AND allowed to
        # pass. Downstream init may raise for other reasons — we care
        # only that the refuse-guard did not.
        with patch(
            "mlpstorage_py.benchmarks.dlio.assert_data_dir_hierarchy_absent"
        ) as mock_guard:
            try:
                TrainingBenchmark(args, logger=MagicMock())
            except ConfigurationError:
                # Some other config check may fire — allowed. What we
                # care about is the refuse-guard invocation.
                pass

            mock_guard.assert_called_once()
            # Guard receives (data_dir, model).
            called_args, _ = mock_guard.call_args
            assert called_args[0] == args.data_dir
            assert called_args[1] == "unet3d"

    def test_datagen_whatif_skips_refuse_guard(self, tmp_path):
        # Whatif is simulation; the refuse-to-overwrite guard must NOT fire.
        (tmp_path / "data" / "unet3d").mkdir(parents=True)
        (tmp_path / "data" / "unet3d" / "prior_shard.npz").write_bytes(b"x")

        args = _training_datagen_args(tmp_path, mode="whatif")
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with patch(
            "mlpstorage_py.benchmarks.dlio.assert_data_dir_hierarchy_absent"
        ) as mock_guard:
            try:
                TrainingBenchmark(args, logger=MagicMock())
            except Exception:
                # Any downstream error is fine — we only assert the
                # whatif exemption for the refuse-guard.
                pass

            mock_guard.assert_not_called()

    def test_run_command_does_not_invoke_datagen_refuse_guard(self, tmp_path):
        # A populated data-dir is required INPUT for `run` — the datagen
        # refuse-to-overwrite guard must not fire on the run path.
        (tmp_path / "data" / "unet3d").mkdir(parents=True)
        (tmp_path / "data" / "unet3d" / "shard.npz").write_bytes(b"x")

        args = _training_datagen_args(tmp_path)
        args.command = "run"
        # verify_benchmark runs for `run` and may raise (or call
        # sys.exit) — either is fine; the refuse-guard invocation is
        # what we assert on.
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with patch(
            "mlpstorage_py.benchmarks.dlio.assert_data_dir_hierarchy_absent"
        ) as mock_guard:
            try:
                TrainingBenchmark(args, logger=MagicMock())
            except (Exception, SystemExit):
                pass
            mock_guard.assert_not_called()


# --------------------------------------------------------------------------- #
# Pre-run guard — supported-model check                                       #
# --------------------------------------------------------------------------- #


class TestSupportedModelWiring:
    """Datagen's ``__init__`` must reject unsupported models per mode."""

    def test_datagen_closed_rejects_cosmoflow(self, tmp_path):
        # cosmoflow was a v2.0 model — no longer in MODELS_CLOSED for v3.0.
        (tmp_path / "data").mkdir(parents=True)

        args = _training_datagen_args(tmp_path, model="cosmoflow")
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with pytest.raises(ConfigurationError) as excinfo:
            TrainingBenchmark(args, logger=MagicMock())

        text = str(excinfo.value)
        assert "cosmoflow" in text
        assert "closed" in text

    def test_datagen_whatif_accepts_cosmoflow(self, tmp_path):
        # Whatif skips the model allowlist — matches the D-29 policy.
        (tmp_path / "data").mkdir(parents=True)

        args = _training_datagen_args(tmp_path, model="cosmoflow", mode="whatif")
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with patch(
            "mlpstorage_py.benchmarks.dlio.validate_supported_model"
        ) as mock_validate:
            try:
                TrainingBenchmark(args, logger=MagicMock())
            except (Exception, SystemExit):
                pass
            # Guard is not invoked in whatif — helper handles it, but
            # the caller short-circuits earlier for symmetry with the
            # refuse-guard.
            mock_validate.assert_not_called()

    def test_datagen_closed_retinanet_and_unet3d_accepted(self, tmp_path):
        # Sanity: both v3.0 supported models pass the guard.
        for model in ("unet3d", "retinanet"):
            data_dir = tmp_path / model / "data"
            data_dir.mkdir(parents=True)
            args = _training_datagen_args(tmp_path / model, model=model)
            from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

            with patch(
                "mlpstorage_py.benchmarks.dlio.validate_supported_model"
            ) as mock_validate:
                try:
                    TrainingBenchmark(args, logger=MagicMock())
                except (Exception, SystemExit):
                    pass
                mock_validate.assert_called_once_with(model, "closed")


# --------------------------------------------------------------------------- #
# Post-run — manifest write + leaf WARN                                       #
# --------------------------------------------------------------------------- #


def _instantiate_datagen_benchmark(tmp_path, *, model="unet3d", mode="closed"):
    """Instantiate a datagen TrainingBenchmark with cluster-info mocked out.

    Callers own creating ``tmp_path/data`` — for post-run tests we want
    an empty data-dir so the refuse-to-overwrite pre-run guard passes.
    """
    args = _training_datagen_args(tmp_path, model=model, mode=mode)
    from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

    logger = MagicMock()
    benchmark = TrainingBenchmark(args, logger=logger)
    # Patch the DLIO invocation path so _run does not actually shell out.
    # The command_method_map for 'datagen' points at execute_command;
    # substitute a no-op so _run reports success without invoking DLIO.
    benchmark.command_method_map["datagen"] = MagicMock(return_value=None)
    return benchmark, logger


class TestPostRunManifestWrite:
    """Manifest is written under ``<data-dir>/<model>/`` after datagen succeeds."""

    def test_datagen_success_writes_manifest_with_dlio_fields(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, _logger = _instantiate_datagen_benchmark(tmp_path)

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert os.path.isfile(manifest_path), (
            f"Expected manifest at {manifest_path}; not found."
        )
        with open(manifest_path) as f:
            data = json.load(f)
        # The three fields the future run-vs-datagen check will consume,
        # pulled from configs/dlio/workload/unet3d_datagen.yaml.
        assert data["model"] == "unet3d"
        assert data["num_files_train"] == 168
        assert data["num_samples_per_file"] == 1
        assert data["record_length_bytes"] == 146600628

    def test_datagen_whatif_skips_manifest_write(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, _logger = _instantiate_datagen_benchmark(
            tmp_path, mode="whatif"
        )

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        # Whatif is simulation — no manifest should land.
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert not os.path.exists(manifest_path)

    def test_datagen_object_storage_skips_local_manifest_write(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, _logger = _instantiate_datagen_benchmark(tmp_path)
        # Simulate the object-storage branch: params_dict was
        # written by _apply_object_storage_params in production.
        benchmark.params_dict["storage.storage_type"] = "s3"

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        # No local manifest for object storage — the file path we
        # would have written to must not exist.
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert not os.path.exists(manifest_path)

    def test_datagen_missing_dlio_field_returns_failure(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # Simulate a broken workload YAML: strip record_length_bytes
        # from combined_params so write_datagen_manifest raises.
        del benchmark.combined_params["dataset"]["record_length_bytes"]

        result = benchmark._run()

        assert result == EXIT_CODE.FAILURE
        # An error message must surface the missing key — silent
        # failure would be worse than the original bug.
        error_msgs = [
            call.args[0]
            for call in logger.error.call_args_list
            if call.args
        ]
        assert any(
            "record_length_bytes" in m for m in error_msgs
        ), f"Expected 'record_length_bytes' in error logs; got {error_msgs!r}"


class TestPostRunLeafWarn:
    """Missing leaf files produce WARN log entries, not FAILURE."""

    def test_missing_leaf_files_emit_warnings(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # The result_dir was reserved during __init__ (empty dir —
        # DLIO would populate it, but our mocked command_method_map
        # never runs). validate_datagen_leaf should report every
        # required file/folder as missing.

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        warn_msgs = [
            call.args[0]
            for call in logger.warning.call_args_list
            if call.args
        ]
        # Sanity: at least the four required files + dlio_config folder
        # should surface as WARN. Assert on a couple of representative
        # anchors rather than the full list to avoid over-pinning.
        joined = "\n".join(warn_msgs)
        assert "stdout" in joined, joined
        assert "dlio.log" in joined, joined
        assert "dlio_config" in joined, joined

    def test_healthy_leaf_produces_no_warn(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # Simulate DLIO's on-disk output: populate the reserved
        # result_dir with all required files.
        leaf = benchmark.run_result_output
        for name in (
            "training_datagen.stdout.log",
            "training_datagen.stderr.log",
            "dlio.log",
        ):
            open(os.path.join(leaf, name), "w").close()
        # The mlpstorage-injected metadata name is timestamped;
        # os.path.basename(leaf) is the datetime segment used in
        # the metadata filename in production.
        ts = os.path.basename(leaf)
        open(os.path.join(leaf, f"training_{ts}_metadata.json"), "w").close()
        dlio_cfg = os.path.join(leaf, "dlio_config")
        os.makedirs(dlio_cfg)
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            open(os.path.join(dlio_cfg, name), "w").close()

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        warn_msgs = [
            call.args[0]
            for call in logger.warning.call_args_list
            if call.args and "Datagen output leaf" in call.args[0]
        ]
        assert warn_msgs == [], (
            f"Unexpected leaf-incomplete warnings on healthy leaf: {warn_msgs}"
        )
