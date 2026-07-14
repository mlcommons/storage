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
    # substitute a stub that sets _last_command_rc=0 (matching a
    # successful DLIO invocation) so the storage#744 rc gate lets the
    # post-run manifest write proceed.
    def _fake_execute_command():
        benchmark._last_command_rc = 0
    benchmark.command_method_map["datagen"] = MagicMock(
        side_effect=_fake_execute_command
    )
    return benchmark, logger


def _populate_healthy_datagen_leaf(benchmark):
    """Populate the reserved leaf with the artifacts DLIO would produce.

    Mirrors what production DLIO writes so ``validate_datagen_leaf``
    returns an empty list and the storage#744 leaf-completeness gate
    admits the manifest write.
    """
    leaf = benchmark.run_result_output
    for name in (
        "training_datagen.stdout.log",
        "training_datagen.stderr.log",
        "dlio.log",
    ):
        open(os.path.join(leaf, name), "w").close()
    # The mlpstorage-injected metadata name is timestamped; the leaf's
    # basename is the datetime segment used in the metadata filename.
    ts = os.path.basename(leaf)
    open(os.path.join(leaf, f"training_{ts}_metadata.json"), "w").close()
    dlio_cfg = os.path.join(leaf, "dlio_config")
    os.makedirs(dlio_cfg)
    for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        open(os.path.join(dlio_cfg, name), "w").close()


class TestPostRunManifestWrite:
    """Manifest is written under ``<data-dir>/<model>/`` after datagen succeeds."""

    def test_datagen_success_writes_manifest_with_dlio_fields(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, _logger = _instantiate_datagen_benchmark(tmp_path)
        _populate_healthy_datagen_leaf(benchmark)

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

    def test_datagen_missing_dlio_field_returns_failure(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        _populate_healthy_datagen_leaf(benchmark)
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


class TestPostRunLeafGate:
    """storage#744: missing leaf artifacts fail the run — no manifest written.

    Prior to storage#744 the leaf-presence check only WARNed; the manifest
    was written even when DLIO had failed to produce required artifacts,
    which let downstream steps treat an incomplete dataset as valid.
    """

    def test_missing_leaf_files_fail_and_skip_manifest(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # The result_dir was reserved during __init__ (empty dir —
        # DLIO would populate it, but our fake execute_command never
        # invokes DLIO). validate_datagen_leaf should report every
        # required file/folder as missing → FAILURE, no manifest.

        result = benchmark._run()

        assert result == EXIT_CODE.FAILURE
        error_msgs = [
            call.args[0]
            for call in logger.error.call_args_list
            if call.args
        ]
        joined = "\n".join(error_msgs)
        assert "stdout" in joined, joined
        assert "dlio.log" in joined, joined
        assert "dlio_config" in joined, joined
        # The summary line names the incomplete dataset the user must
        # regenerate — the whole point of the loud failure.
        assert any(
            "must be regenerated" in m for m in error_msgs
        ), f"Expected 'must be regenerated' summary; got {error_msgs!r}"
        # And the manifest MUST NOT exist — that's the storage#744 bug.
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert not os.path.exists(manifest_path), (
            f"Manifest written despite missing leaf artifacts: {manifest_path}"
        )

    def test_healthy_leaf_produces_no_error(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        _populate_healthy_datagen_leaf(benchmark)

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        error_msgs = [
            call.args[0]
            for call in logger.error.call_args_list
            if call.args and "Datagen output leaf" in call.args[0]
        ]
        assert error_msgs == [], (
            f"Unexpected leaf-incomplete errors on healthy leaf: {error_msgs}"
        )

    def test_dlio_only_leaf_succeeds_without_pre_seeded_metadata(self, tmp_path):
        """storage#767: production DLIO never writes the mlpstorage metadata
        file; that file is written by ``Benchmark.write_metadata()`` in
        main.py's ``finally`` block, i.e. AFTER ``_run`` returns. Before
        the fix, ``_post_datagen_actions`` ran the leaf check first and
        false-positived on every real datagen run with
        ``ERROR: Datagen output leaf incomplete: training_.*_metadata.json``.
        This test seeds only the four artifacts DLIO actually produces and
        asserts SUCCESS + a written manifest + the metadata file present
        on disk after ``_run``.
        """
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # Seed only what production DLIO writes — deliberately NO metadata.
        leaf = benchmark.run_result_output
        for name in (
            "training_datagen.stdout.log",
            "training_datagen.stderr.log",
            "dlio.log",
        ):
            open(os.path.join(leaf, name), "w").close()
        dlio_cfg = os.path.join(leaf, "dlio_config")
        os.makedirs(dlio_cfg)
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            open(os.path.join(dlio_cfg, name), "w").close()

        result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        error_msgs = [
            call.args[0]
            for call in logger.error.call_args_list
            if call.args and "Datagen output leaf" in call.args[0]
        ]
        assert error_msgs == [], (
            f"storage#767 regression: leaf check false-positived on "
            f"metadata file that _post_datagen_actions is now responsible "
            f"for writing pre-check. Errors: {error_msgs}"
        )
        assert os.path.isfile(benchmark.metadata_file_path), (
            f"Expected metadata file {benchmark.metadata_file_path} to "
            f"exist after _run — _post_datagen_actions must write it "
            f"before the leaf check."
        )
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert os.path.isfile(manifest_path), (
            f"Expected manifest at {manifest_path}; not found."
        )


class TestDLIOFailureGate:
    """storage#744: non-zero DLIO exit code fails the run — no manifest written."""

    def test_dlio_nonzero_rc_skips_manifest_and_returns_failure(self, tmp_path):
        (tmp_path / "data").mkdir()
        benchmark, logger = _instantiate_datagen_benchmark(tmp_path)
        # Even with a healthy leaf on disk, a non-zero DLIO rc must
        # block the manifest — the leaf may look complete but the run
        # itself failed (killed process, MPI bad-termination, etc.).
        _populate_healthy_datagen_leaf(benchmark)
        # Override the fake execute_command to simulate mpiexec
        # returning a non-zero exit code (as it does when a rank is
        # killed with EXIT CODE: 9).
        def _fake_execute_failure():
            benchmark._last_command_rc = 9
        benchmark.command_method_map["datagen"] = MagicMock(
            side_effect=_fake_execute_failure
        )

        result = benchmark._run()

        assert result == EXIT_CODE.FAILURE
        error_msgs = [
            call.args[0]
            for call in logger.error.call_args_list
            if call.args
        ]
        assert any(
            "non-zero status" in m and "9" in m for m in error_msgs
        ), f"Expected rc-9 error message; got {error_msgs!r}"
        assert any(
            "must be regenerated" in m for m in error_msgs
        ), f"Expected 'must be regenerated' summary; got {error_msgs!r}"
        # And the manifest MUST NOT exist even though the on-disk leaf
        # was populated — the run itself failed.
        manifest_path = os.path.join(
            benchmark.args.data_dir, "unet3d", DATAGEN_MANIFEST_FILENAME
        )
        assert not os.path.exists(manifest_path), (
            f"Manifest written despite non-zero DLIO rc: {manifest_path}"
        )


# --------------------------------------------------------------------------- #
# Object-storage parity — TrainingBenchmark invokes the S3 branches           #
# --------------------------------------------------------------------------- #


def _apply_object_storage_stub(self):
    """Stub for ``_apply_object_storage_params``: set the s3 params directly.

    In production the real method reads ``BUCKET`` / ``STORAGE_LIBRARY``
    from the environment (via ``.env`` loading) and mutates
    ``params_dict``. Under test we bypass the .env dance and just
    inject the two keys the downstream storage-scheme-consistency
    check compares — that matches the production shape without
    needing real credentials.
    """
    self.params_dict['storage.storage_type'] = 's3'


@pytest.fixture
def _object_storage_env_stubs():
    """Stub the .env-driven object-storage wiring so tests can pass an s3:// URI.

    Two things need bypassing:

    - ``_apply_object_storage_params`` reads env vars → replace with a
      stub that just injects ``storage.storage_type='s3'``.
    - ``_check_storage_scheme_consistency`` verifies s3 storage_type
      pairs with an s3:// dataset URI; the real check would pass here
      but pulls in production plumbing we don't need in the test.
    """
    with patch(
        "mlpstorage_py.benchmarks.dlio.DLIOBenchmark._apply_object_storage_params",
        _apply_object_storage_stub,
    ), patch(
        "mlpstorage_py.benchmarks.dlio.DLIOBenchmark._check_storage_scheme_consistency",
        lambda self: None,
    ):
        yield


class TestTrainingBenchmarkObjectStorageWiring:
    """The datagen wiring must fire the guards for object-storage --data-dir too.

    Prior to this test class, both the pre-run refuse guard and the
    post-run manifest write were gated on ``storage_type == 'local'``
    which made them silent no-ops on S3. This class pins the parity:
    an ``s3://`` --data-dir must:

    - Invoke ``assert_data_dir_hierarchy_absent`` (which internally
      dispatches to ``s3dlio.list``, mocked here) BEFORE add_datadir_param.
    - Invoke ``write_datagen_manifest`` (which internally dispatches
      to ``s3dlio.put_bytes``, mocked here) AFTER DLIO returns success.
    """

    def test_s3_datagen_invokes_refuse_and_manifest_write(
        self, tmp_path, _object_storage_env_stubs
    ):
        # Point --data-dir at an s3 URI. --results-dir stays local
        # because submission artifacts always land locally.
        args = _training_datagen_args(tmp_path)
        args.data_dir = "s3://mybucket/mytraining"

        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        # Patch s3dlio at the module boundary
        # (mlpstorage_py.rules.datagen_hierarchy.s3dlio). Empty LIST
        # → refuse-guard passes; put_bytes mock captures the manifest
        # PUT arguments for verification.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.list.return_value = []

            benchmark = TrainingBenchmark(args, logger=MagicMock())
            # LIST fired during __init__ pre-run guard.
            assert mock_s3dlio.list.call_count == 1
            (list_uri,), list_kwargs = mock_s3dlio.list.call_args
            assert list_uri.startswith("s3://mybucket/mytraining")
            assert "unet3d" in list_uri
            assert list_kwargs.get("recursive") is False

            # Replace DLIO invocation with a stub that sets rc=0 (the
            # storage#744 gate). Populate the local leaf with the
            # DLIO-produced artifacts so the leaf-completeness gate
            # also admits the write.
            def _fake_execute_command():
                benchmark._last_command_rc = 0
            benchmark.command_method_map["datagen"] = MagicMock(
                side_effect=_fake_execute_command
            )
            _populate_healthy_datagen_leaf(benchmark)

            result = benchmark._run()

        assert result == EXIT_CODE.SUCCESS
        # PUT fired during post-run manifest write.
        mock_s3dlio.put_bytes.assert_called_once()
        (put_uri, payload), _ = mock_s3dlio.put_bytes.call_args
        assert put_uri == (
            "s3://mybucket/mytraining/unet3d/"
            ".mlps-datagen-manifest.json"
        )
        assert isinstance(payload, (bytes, bytearray))

    def test_s3_datagen_refuses_when_prefix_non_empty(
        self, tmp_path, _object_storage_env_stubs
    ):
        args = _training_datagen_args(tmp_path)
        args.data_dir = "s3://mybucket/mytraining"

        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            # LIST returns a non-empty result → refuse-guard raises.
            mock_s3dlio.list.return_value = [
                "s3://mybucket/mytraining/unet3d/train/shard_000.npz"
            ]
            with pytest.raises(ConfigurationError) as excinfo:
                TrainingBenchmark(args, logger=MagicMock())

        text = str(excinfo.value)
        assert "unet3d" in text
        assert "s3://mybucket/mytraining" in text
        # And put_bytes MUST NOT have been called — refuse fires
        # before any write path.
        mock_s3dlio.put_bytes.assert_not_called()

    def test_s3_datagen_whatif_skips_both_s3_calls(
        self, tmp_path, _object_storage_env_stubs
    ):
        args = _training_datagen_args(tmp_path, mode="whatif")
        args.data_dir = "s3://mybucket/mytraining"

        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            benchmark = TrainingBenchmark(args, logger=MagicMock())
            benchmark.command_method_map["datagen"] = MagicMock(
                return_value=None
            )
            benchmark._run()

        # Whatif skips both the pre-run guard and the post-run
        # manifest write per the D-29 policy.
        mock_s3dlio.list.assert_not_called()
        mock_s3dlio.put_bytes.assert_not_called()
