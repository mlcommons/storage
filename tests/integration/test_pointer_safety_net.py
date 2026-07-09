"""#725 Bug 2 safety-net: Benchmark.__init__ writes .mlps-code-image to the
ACTUAL reserved leaf.

Field report showed a completed checkpointing run whose reserved leaf held
no ``.mlps-code-image`` pointer, so CHECK-01/CHECK-03 flagged the fresh pool
image as orphaned. The capture-time pointer write inside
``capture_or_verify_code_image`` targets a leaf computed via a
``SimpleNamespace`` shim + ``DATETIME_STR`` — a path that can diverge from
Benchmark's own ``generate_output_location(self, self.run_datetime)`` under
conditions the static repro could not surface. The fix stashes
``args._validated_pool_hash`` during capture and has ``Benchmark.__init__``
re-emit the pointer at ``self.run_result_output`` after
``_reserve_run_directory`` returns.

These tests bypass ``capture_or_verify_code_image`` entirely — they set
``args._validated_pool_hash`` directly and instantiate a Benchmark subclass.
Without the safety net the reserved leaf holds no pointer (RED).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.sample_data import create_sample_benchmark_args


_LIVE_HASH = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _bypass_capacity_gate():
    with patch(
        "mlpstorage_py.benchmarks.base.Benchmark._pre_execution_gate",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _mock_cluster_information():
    with patch("mlpstorage_py.benchmarks.base.ClusterInformation") as mock_ci:
        mock_ci.return_value = MagicMock()
        mock_ci.return_value.total_memory_bytes = 256 * 1024**3
        mock_ci.return_value.host_info_list = []
        yield mock_ci


def _training_args(tmp_path):
    args = create_sample_benchmark_args(
        benchmark_type="training",
        command="run",
        model="unet3d",
        accelerator_type="h100",
        num_accelerators=8,
        client_host_memory_in_gb=256,
        hosts=["127.0.0.1"],
        data_dir=str(tmp_path / "data"),
    )
    args.mode = "closed"
    args.results_dir = str(tmp_path / "results")
    args.dry_run = True
    args.what_if = False
    return args


class TestPointerSafetyNet:
    """Benchmark.__init__ writes the pointer to the actual reserved leaf."""

    def test_reserved_leaf_receives_pointer_from_stashed_hash(self, tmp_path):
        """With ``args._validated_pool_hash`` set (as capture stashes it),
        the reserved leaf must hold ``.mlps-code-image`` on return from
        ``Benchmark.__init__`` — even when capture's own shim-based write
        never fires.
        """
        (tmp_path / "results").mkdir()
        args = _training_args(tmp_path)
        args._validated_pool_hash = _LIVE_HASH

        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        try:
            benchmark = TrainingBenchmark(args, logger=MagicMock())
        except BaseException:
            # Downstream init may raise for unrelated reasons in a
            # test-only environment (no MPI, no datagen tree). What
            # matters is whether the pointer landed before that raise.
            reserved = None
            for root, _dirs, files in os.walk(tmp_path / "results"):
                if ".mlps-code-image" in files:
                    reserved = root
                    break
            assert reserved is not None, (
                "no .mlps-code-image pointer under results/ after "
                "Benchmark.__init__ raised"
            )
            return

        pointer = os.path.join(benchmark.run_result_output, ".mlps-code-image")
        assert os.path.isfile(pointer), (
            f".mlps-code-image missing at reserved leaf "
            f"{benchmark.run_result_output!r} — safety net did not fire"
        )
        content = open(pointer).read().strip()
        assert content == f"md5-tree-v2:{_LIVE_HASH}", (
            f"pointer content mismatch: {content!r}"
        )

    def test_missing_stashed_hash_leaves_pointer_absent(self, tmp_path):
        """When ``args._validated_pool_hash`` is unset (unit-test paths,
        non-submission modes gated off in capture at D-10), Benchmark
        must not synthesize a pointer. The safety net is opt-in.
        """
        (tmp_path / "results").mkdir()
        args = _training_args(tmp_path)
        # NOTE: no _validated_pool_hash

        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        try:
            benchmark = TrainingBenchmark(args, logger=MagicMock())
            leaf = benchmark.run_result_output
        except BaseException:
            # Downstream init may fail before we reach the assertion —
            # in that case just confirm no pointer appeared anywhere.
            for root, _dirs, files in os.walk(tmp_path / "results"):
                assert ".mlps-code-image" not in files, (
                    f"pointer written at {root!r} without stashed hash"
                )
            return

        pointer = os.path.join(leaf, ".mlps-code-image")
        assert not os.path.exists(pointer), (
            f"pointer must not appear without _validated_pool_hash: {pointer}"
        )
