"""Integration tests: datagen_hierarchy guards wired into TrainingBenchmark.

The unit tests in ``tests/unit/test_datagen_hierarchy.py`` prove the four
helpers in ``mlpstorage_py.rules.datagen_hierarchy`` behave correctly in
isolation. This file proves that the training benchmark's
``__init__`` actually calls them at the right points and only for the
right command/mode combinations:

    - Pre-run guards fire for ``command == "datagen"``.
    - Whatif mode skips all guards (matches D-29 reportgen policy and
      the user's "no significant validation checking" direction).
    - Non-datagen commands (``run``, ``configview``, ``datasize``) do
      not invoke datagen-specific guards — a populated ``--data-dir``
      is expected input for ``run``, not a refuse-to-overwrite signal.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from mlpstorage_py.errors import ConfigurationError
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
                except Exception:
                    pass
                mock_validate.assert_called_once_with(model, "closed")
