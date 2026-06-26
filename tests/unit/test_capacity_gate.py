"""Unit tests for Phase 5 / Plan 05-03 CAP-01 capacity gate.

Locks the four-field-message contract from REQUIREMENTS.md CAP-01 + D-45:

    CAP-01: insufficient disk space at <destination_path>
      available_bytes: <int>
      required_bytes:  <int>
      deficit:         <int>

And the template-method gate ordering from Plan 05-03:

    Benchmark.run():
        _collect_cluster_start()
        _pre_execution_gate()             <-- Slice 3 (CAP-01)  / Slice 4 (CAP-02)
        write_systemname_yaml(...)        <-- Slice 2 (LIFE-02) inside its try/except

Test discipline:
- A6 KVCache 1x lock: required_bytes is int(total_cache_mb * 1024 * 1024), NOT *2.
- A7 Checkpointing destination join: os.path.join(args.checkpoint_folder, args.model).
- A8 Remote-backend escape hatch: VectorDB returns None destination on milvus URIs.
- SC#6 silence lock: happy path returns None and emits ZERO logger calls.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# Stub heavy deps the benchmark imports expect (pre-existing dev-env psutil gap
# documented in STATE.md Deferred Items). Use importlib.util.find_spec — checking
# sys.modules alone would install a MagicMock for a perfectly importable module
# that just hasn't been imported yet, which then poisons later test collections
# (e.g. test_parquet_reader, test_dlio_object_storage) by causing find_spec to
# raise ValueError on the Mock's __spec__. Matches the safe pattern in
# tests/unit/test_benchmarks_kvcache.py.
import importlib.util as _ilu
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.benchmarks.capacity_gate import check_capacity_4field
from mlpstorage_py.errors import ErrorCode, FileSystemError


# =============================================================================
# TestCheckCapacity4Field — the 4-field message + parent-walk + error codes
# =============================================================================


class TestCheckCapacity4Field:
    """Direct unit tests for check_capacity_4field()."""

    def test_happy_path_returns_none_silent(self, tmp_path):
        """SC#6 silence lock: zero logger calls on the success path."""
        logger = MagicMock()
        # 1 byte against any tmp_path with megabytes free should pass.
        result = check_capacity_4field(str(tmp_path), 1, logger)
        assert result is None
        logger.info.assert_not_called()
        logger.warning.assert_not_called()
        logger.error.assert_not_called()
        logger.debug.assert_not_called()

    def test_insufficient_space_raises_filesystem_error(self, tmp_path):
        # Require 10^20 bytes (~100 ZB); no tmp filesystem has that.
        with pytest.raises(FileSystemError):
            check_capacity_4field(str(tmp_path), 10**20, None)

    def test_insufficient_space_message_contains_destination_path(self, tmp_path):
        with pytest.raises(FileSystemError) as exc_info:
            check_capacity_4field(str(tmp_path), 10**20, None)
        assert str(tmp_path) in str(exc_info.value)

    def test_insufficient_space_message_contains_available_bytes(self, tmp_path):
        with pytest.raises(FileSystemError) as exc_info:
            check_capacity_4field(str(tmp_path), 10**20, None)
        assert "available_bytes:" in str(exc_info.value)

    def test_insufficient_space_message_contains_required_bytes(self, tmp_path):
        with pytest.raises(FileSystemError) as exc_info:
            check_capacity_4field(str(tmp_path), 10**20, None)
        assert "required_bytes:" in str(exc_info.value)

    def test_insufficient_space_message_contains_deficit(self, tmp_path):
        with pytest.raises(FileSystemError) as exc_info:
            check_capacity_4field(str(tmp_path), 10**20, None)
        assert "deficit:" in str(exc_info.value)

    def test_insufficient_space_uses_fs_disk_full_code(self, tmp_path):
        with pytest.raises(FileSystemError) as exc_info:
            check_capacity_4field(str(tmp_path), 10**20, None)
        assert exc_info.value.error.code == ErrorCode.FS_DISK_FULL

    def test_no_valid_parent_raises_fs_path_not_found(self):
        """When dirname(p) == p (root) and the root does not exist,
        the parent-walk terminates and we raise FS_PATH_NOT_FOUND.

        Construct this by mocking os.path.exists to ALWAYS return False so
        the parent walk terminates at '/' (where dirname('/') == '/').
        """
        with patch("mlpstorage_py.benchmarks.capacity_gate.os.path.exists", return_value=False):
            with pytest.raises(FileSystemError) as exc_info:
                check_capacity_4field("/nonexistent/very/deep/path", 1, None)
        assert exc_info.value.error.code == ErrorCode.FS_PATH_NOT_FOUND

    def test_statvfs_oserror_raises_fs_permission_denied(self, tmp_path):
        with patch(
            "mlpstorage_py.benchmarks.capacity_gate.os.statvfs",
            side_effect=OSError("EACCES"),
        ):
            with pytest.raises(FileSystemError) as exc_info:
                check_capacity_4field(str(tmp_path), 1, None)
        assert exc_info.value.error.code == ErrorCode.FS_PERMISSION_DENIED

    def test_parent_walk_finds_existing_parent_when_target_does_not_exist(self, tmp_path):
        """Pitfall 5: the gate must reach the existing parent (tmp_path) and
        statvfs against it — NOT raise FS_PATH_NOT_FOUND just because the
        leaf 'does/not/exist' is missing.
        """
        nonexistent_leaf = str(tmp_path / "does" / "not" / "exist")
        # Should NOT raise (the parent exists with abundant space; 1 byte required).
        result = check_capacity_4field(nonexistent_leaf, 1, None)
        assert result is None


# =============================================================================
# TestPreExecutionGateBaseClass — Benchmark._pre_execution_gate template method
# =============================================================================


def _make_mock_benchmark(destination, required_bytes, logger=None):
    """Construct a bare Benchmark-like object that exposes only the
    surface _pre_execution_gate touches. We bypass __init__ because the
    full Benchmark.__init__ has many side effects (run-dir reservation,
    code-image capture, etc.) unrelated to the gate's contract.

    Plan 05-04 (CAP-02 wiring) note: _pre_execution_gate now also reads
    self.args.hosts and self._run_uuid to invoke run_shared_fs_probe.
    We provide an empty hosts list by default so the probe takes its
    SC#8 single-host no-op short-circuit; tests that care about the
    CAP-02 path can override args.hosts and patch run_shared_fs_probe.
    """
    bm = MagicMock(spec=Benchmark)
    bm._capacity_gate_destination = MagicMock(return_value=destination)
    bm.required_bytes_for_capacity_gate = MagicMock(return_value=required_bytes)
    bm.logger = logger or MagicMock()
    bm.args = SimpleNamespace(hosts=[], mpi_bin=None,
                              allow_run_as_root=False, ssh_username=None)
    bm._run_uuid = "test-uuid-mock"
    # Bind the real method to the mock so it actually executes.
    bm._pre_execution_gate = Benchmark._pre_execution_gate.__get__(bm, MagicMock)
    return bm


class TestPreExecutionGateBaseClass:
    """Tests for Benchmark._pre_execution_gate() — the template method."""

    def test_pre_execution_gate_calls_required_bytes_then_capacity_check(self, tmp_path):
        bm = _make_mock_benchmark(str(tmp_path), 1)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ) as mock_check:
            bm._pre_execution_gate()
        bm._capacity_gate_destination.assert_called_once()
        bm.required_bytes_for_capacity_gate.assert_called_once()
        mock_check.assert_called_once_with(str(tmp_path), 1, bm.logger)

    def test_pre_execution_gate_skips_check_when_destination_is_none_remote_backend(self, tmp_path):
        """A8 escape hatch: a None destination means a remote-only backend.
        Log info and SKIP the local statvfs (which would be meaningless).
        """
        logger = MagicMock()
        bm = _make_mock_benchmark(None, 999, logger=logger)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ) as mock_check:
            bm._pre_execution_gate()
        mock_check.assert_not_called()
        # An info-log explaining the skip is expected.
        info_calls = logger.info.call_args_list
        assert any(
            "CAP-01 skipped" in (args[0] if args else "")
            for args, _ in info_calls
        )

    def test_pre_execution_gate_propagates_filesystem_error_from_check(self, tmp_path):
        bm = _make_mock_benchmark(str(tmp_path), 10**20)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field",
            side_effect=FileSystemError(
                "boom",
                path=str(tmp_path),
                operation="cap01-check",
                code=ErrorCode.FS_DISK_FULL,
            ),
        ):
            with pytest.raises(FileSystemError):
                bm._pre_execution_gate()

    def test_base_class_required_bytes_raises_not_implemented(self):
        """A bare Benchmark subclass that does NOT override
        required_bytes_for_capacity_gate must raise NotImplementedError
        with the class name in the message.
        """
        # Use the unbound method to bypass having to construct a real instance.
        fake_self = MagicMock(spec=Benchmark)
        type(fake_self).__name__ = "FakeBM"
        with pytest.raises(NotImplementedError) as exc_info:
            Benchmark.required_bytes_for_capacity_gate(fake_self)
        # The error message should name the class.
        assert "FakeBM" in str(exc_info.value) or "required_bytes_for_capacity_gate" in str(exc_info.value)

    def test_base_class_capacity_gate_destination_raises_not_implemented(self):
        fake_self = MagicMock(spec=Benchmark)
        type(fake_self).__name__ = "FakeBM"
        with pytest.raises(NotImplementedError) as exc_info:
            Benchmark._capacity_gate_destination(fake_self)
        assert "FakeBM" in str(exc_info.value) or "_capacity_gate_destination" in str(exc_info.value)


# =============================================================================
# TestRunInvokesPreExecutionGate — Benchmark.run() call-site ordering
# =============================================================================


class TestRunInvokesPreExecutionGate:
    """Lock ordering: _collect_cluster_start -> _pre_execution_gate -> write_systemname_yaml."""

    def test_run_calls_pre_execution_gate_after_collect_cluster_start_before_write(self):
        """Order is enforced by reading run()'s source positionally —
        a call_order on mocks. We construct a minimal benchmark via
        a MagicMock(spec=Benchmark) and call run() with the relevant
        helpers replaced by MagicMocks that record the parent's
        call order in a shared list.
        """
        call_order: list = []

        def rec(name):
            def _f(*a, **kw):
                call_order.append(name)
            return _f

        bm = MagicMock(spec=Benchmark)
        bm.logger = MagicMock()
        bm._validate_environment.side_effect = rec("validate")
        bm._collect_cluster_start.side_effect = rec("collect_start")
        bm._pre_execution_gate.side_effect = rec("gate")
        bm._start_timeseries_collection.side_effect = rec("ts_start")
        bm._stop_timeseries_collection.side_effect = rec("ts_stop")
        bm._collect_cluster_end.side_effect = rec("collect_end")
        bm._run.side_effect = lambda: (call_order.append("run"), 0)[-1]
        bm._cluster_info_start = None
        bm.args = SimpleNamespace(command="run")

        with patch("mlpstorage_py.benchmarks.base.write_systemname_yaml") as mock_write, \
             patch("mlpstorage_py.benchmarks.base.create_stage_progress") as mock_progress:
            mock_write.side_effect = rec("write_yaml")
            # Make the stage-progress context manager a no-op that yields
            # a callable advance_stage.
            mock_progress.return_value.__enter__.return_value = lambda: None
            mock_progress.return_value.__exit__.return_value = False
            Benchmark.run(bm)

        # Order constraints: gate AFTER collect_start, BEFORE write_yaml.
        assert call_order.index("collect_start") < call_order.index("gate")
        assert call_order.index("gate") < call_order.index("write_yaml")

    def test_pre_execution_gate_failure_aborts_before_write_systemname_yaml(self):
        """If _pre_execution_gate raises, run() must NOT reach write_systemname_yaml."""
        bm = MagicMock(spec=Benchmark)
        bm.logger = MagicMock()
        bm._pre_execution_gate.side_effect = FileSystemError(
            "starved",
            path="/data",
            operation="cap01-check",
            code=ErrorCode.FS_DISK_FULL,
        )
        bm._cluster_info_start = None
        bm.args = SimpleNamespace(command="run")

        with patch("mlpstorage_py.benchmarks.base.write_systemname_yaml") as mock_write, \
             patch("mlpstorage_py.benchmarks.base.create_stage_progress") as mock_progress:
            mock_progress.return_value.__enter__.return_value = lambda: None
            mock_progress.return_value.__exit__.return_value = False
            with pytest.raises(FileSystemError):
                Benchmark.run(bm)

        mock_write.assert_not_called()


# =============================================================================
# Per-subclass tests — A6 / A7 / A8 locks
# =============================================================================


class TestTrainingBenchmarkRequiredBytes:
    """A7 destination + calculate_training_data_size delegation."""

    def test_returns_third_tuple_element_of_calculate_training_data_size(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        bm = MagicMock(spec=TrainingBenchmark)
        bm.args = SimpleNamespace(data_dir="/data")
        bm.cluster_information = MagicMock()
        bm.combined_params = {"dataset": {}, "reader": {}}
        bm.logger = MagicMock()

        with patch(
            "mlpstorage_py.benchmarks.dlio.calculate_training_data_size",
            return_value=(100, 5, 12_345_678_900),
        ):
            result = TrainingBenchmark.required_bytes_for_capacity_gate(bm)
        assert result == 12_345_678_900

    def test_destination_is_args_data_dir(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        bm = MagicMock(spec=TrainingBenchmark)
        bm.args = SimpleNamespace(data_dir="/data/foo")
        assert TrainingBenchmark._capacity_gate_destination(bm) == "/data/foo"

    def test_datasize_invokes_pre_execution_gate_before_calculate_training_data_size(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        order = []
        bm = MagicMock(spec=TrainingBenchmark)
        bm._pre_execution_gate = MagicMock(side_effect=lambda: order.append("gate"))
        bm.args = SimpleNamespace(data_dir="/d", hosts=None, exec_type="local",
                                  num_processes=1, results_dir="/r", mode="closed",
                                  model="unet3d")
        bm.cluster_information = MagicMock()
        bm.combined_params = {"dataset": {}, "reader": {}}
        bm.params_dict = {}
        bm.logger = MagicMock()

        def fake_calc(*a, **kw):
            order.append("calc")
            return (10, 1, 1024)
        bm.generate_datagen_benchmark_command = MagicMock(return_value="cmd")

        with patch(
            "mlpstorage_py.benchmarks.dlio.calculate_training_data_size",
            side_effect=fake_calc,
        ):
            TrainingBenchmark.datasize(bm)
        assert order == ["gate", "calc"]

    def test_required_bytes_lazy_collects_cluster_information_when_missing(self):
        """HARDEN-01 regression guard: locks the 754763a + 29f1062 fix in place (Phase 5.1).

        Phase 5.1 / Gap fix: datagen path doesn't pre-collect cluster info
        (Benchmark._collect_cluster_start short-circuits for
        args.command in ('datagen','configview')), yet
        Benchmark.run() still fires _pre_execution_gate after that
        short-circuit. required_bytes_for_capacity_gate must therefore
        lazy-collect cluster_information instead of assuming __init__ /
        _collect_cluster_start already set it. Repro for AttributeError:
        '_pre_execution_gate' raised
        \"'TrainingBenchmark' object has no attribute 'cluster_information'\"
        on `mlpstorage closed training unet3d datagen ...`.
        """
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        bm = MagicMock(spec=TrainingBenchmark)
        bm.args = SimpleNamespace(data_dir="/data")
        # Critical: bm starts WITHOUT cluster_information set (datagen flow).
        # spec=TrainingBenchmark inheritance makes the attribute missing rather
        # than auto-mocked. Use del to be defensive in case mock created it.
        try:
            del bm.cluster_information
        except AttributeError:
            pass
        bm.combined_params = {"dataset": {}, "reader": {}}
        bm.logger = MagicMock()

        collected = MagicMock(name="cluster_info_collected")
        bm.accumulate_host_info = MagicMock(return_value=collected)

        with patch(
            "mlpstorage_py.benchmarks.dlio.calculate_training_data_size",
            return_value=(100, 5, 9_999_999),
        ) as mock_calc:
            result = TrainingBenchmark.required_bytes_for_capacity_gate(bm)

        # Bug fix: lazy collection happened
        bm.accumulate_host_info.assert_called_once_with(bm.args)
        # And the collected info was threaded into calculate_training_data_size
        # (not the missing attribute that triggered the AttributeError)
        called_args, called_kwargs = mock_calc.call_args
        assert collected in called_args or collected in called_kwargs.values(), (
            "calculate_training_data_size must receive the lazy-collected "
            "ClusterInformation, not be invoked with a missing attribute."
        )
        assert result == 9_999_999

    def test_required_bytes_uses_existing_cluster_information_when_already_set(self):
        """Run-command path already pre-collected cluster info via
        Benchmark._collect_cluster_start. The lazy-collect must NOT
        re-collect or overwrite — that would double-MPI on every run."""
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        bm = MagicMock(spec=TrainingBenchmark)
        bm.args = SimpleNamespace(data_dir="/data")
        existing = MagicMock(name="cluster_info_existing")
        bm.cluster_information = existing
        bm.combined_params = {"dataset": {}, "reader": {}}
        bm.logger = MagicMock()
        bm.accumulate_host_info = MagicMock()

        with patch(
            "mlpstorage_py.benchmarks.dlio.calculate_training_data_size",
            return_value=(100, 5, 1234),
        ) as mock_calc:
            TrainingBenchmark.required_bytes_for_capacity_gate(bm)

        bm.accumulate_host_info.assert_not_called()
        called_args, called_kwargs = mock_calc.call_args
        assert existing in called_args or existing in called_kwargs.values()

    def test_required_bytes_returns_zero_when_lazy_collect_cannot_determine_memory(self):
        """When MPI collection fails AND args lacks client_host_memory_in_gb
        (the realistic datagen-from-CLI flow on a dev/submitter box),
        accumulate_host_info raises AttributeError on the CLI-fallback path.
        The CAP-01 gate must degrade gracefully — log a clear "deferred"
        notice and return 0 so _pre_execution_gate's check_capacity_4field
        becomes a no-op rather than crashing the benchmark.

        Parallels the A8 escape-hatch pattern in VectorDB
        (_capacity_gate_destination returning None).
        """
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

        bm = MagicMock(spec=TrainingBenchmark)
        bm.args = SimpleNamespace(data_dir="/data")  # NO client_host_memory_in_gb
        try:
            del bm.cluster_information
        except AttributeError:
            pass
        bm.combined_params = {"dataset": {}, "reader": {}}
        bm.logger = MagicMock()

        def _raise_missing_arg(_args):
            raise AttributeError(
                "'Namespace' object has no attribute 'client_host_memory_in_gb'"
            )
        bm.accumulate_host_info = MagicMock(side_effect=_raise_missing_arg)

        result = TrainingBenchmark.required_bytes_for_capacity_gate(bm)

        assert result == 0
        # A clear operator-visible notice is required (parallel to A8 escape).
        assert bm.logger.info.called, (
            "Operator must be told CAP-01 was deferred — silent skip violates SC#6 intent."
        )


class TestCheckpointingBenchmarkRequiredBytes:
    """A7 destination join + sum(rank_gb) * GiB * num_checkpoints_write."""

    def test_returns_sum_rank_gb_times_gib_times_num_checkpoints_write(self):
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
        from mlpstorage_py.config import LLM_ALLOWED_VALUES, LLM_SIZE_BY_RANK

        bm = MagicMock(spec=CheckpointingBenchmark)
        bm.args = SimpleNamespace(
            model="llama3-8b",
            num_processes=8,
            num_checkpoints_write=3,
            checkpoint_folder="/cp",
        )
        bm.logger = MagicMock()

        # llama3-8b: ZeroLevel=3 (sharded across all ranks), model=15, optimizer=90
        # rank_gb[i] = (15 + 90) / 8 = 13.125 for each of 8 ranks
        # sum = 105.0; expected = int(105.0 * 1024**3 * 3) = 338368201523 (approx)
        model_gb, optimizer_gb = LLM_SIZE_BY_RANK["llama3-8b"]
        per_rank = (model_gb + optimizer_gb) / 8
        expected = int(per_rank * 8 * 1024**3 * 3)

        result = CheckpointingBenchmark.required_bytes_for_capacity_gate(bm)
        assert result == expected

    def test_destination_is_checkpoint_folder_joined_with_model(self):
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark

        bm = MagicMock(spec=CheckpointingBenchmark)
        bm.args = SimpleNamespace(checkpoint_folder="/cp", model="llama3-8b")
        result = CheckpointingBenchmark._capacity_gate_destination(bm)
        assert result == os.path.join("/cp", "llama3-8b")

    def test_destination_is_none_when_checkpoint_folder_empty(self):
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark

        bm = MagicMock(spec=CheckpointingBenchmark)
        bm.args = SimpleNamespace(checkpoint_folder=None, model="llama3-8b")
        result = CheckpointingBenchmark._capacity_gate_destination(bm)
        assert result is None


class TestVectorDBBenchmarkRequiredBytes:
    """A8 escape hatch + execute_datasize math parity."""

    def test_returns_num_vectors_times_dim_times_4_times_overhead_times_num_shards(self):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        bm = MagicMock(spec=VectorDBBenchmark)
        bm.args = SimpleNamespace(
            num_vectors=1_000_000,
            dimension=768,
            num_shards=2,
        )
        # DISKANN overhead = 1.3
        bm._effective_index_type = MagicMock(return_value="DISKANN")
        expected = int(1_000_000 * 768 * 4 * 1.3 * 2)
        result = VectorDBBenchmark.required_bytes_for_capacity_gate(bm)
        assert result == expected

    def test_local_backend_returns_destination_path(self):
        """Even on local destinations, VectorDB is fundamentally a remote-engine
        benchmark (data lands in the VDB process, not on a local mount the
        benchmark itself controls). Per A8 we return None — let the gate skip.
        """
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        bm = MagicMock(spec=VectorDBBenchmark)
        bm.args = SimpleNamespace(host="localhost")
        # Per A8 contract: VDB destinations are always remote engines.
        result = VectorDBBenchmark._capacity_gate_destination(bm)
        assert result is None

    def test_remote_milvus_backend_returns_none_destination(self):
        """A8 escape hatch — milvus URI explicitly returns None."""
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        bm = MagicMock(spec=VectorDBBenchmark)
        bm.args = SimpleNamespace(host="my-milvus.cluster.local")
        result = VectorDBBenchmark._capacity_gate_destination(bm)
        assert result is None


class TestKVCacheBenchmarkRequiredBytes:
    """A6 1x lock + cache-path destination + internal model table source."""

    def test_returns_total_cache_bytes_at_1x_per_a6(self):
        """A6 KEY LOCK: returns int(total_cache_mb * 1024 * 1024), NOT *2.

        The 2x recommendation at kvcache.py:336 is for performance headroom
        and stays in the user-facing log; CAP-01 enforces the floor.
        """
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        bm = MagicMock(spec=KVCacheBenchmark)
        # Bind the real class-level tables so .get() returns real values.
        bm._MODEL_CACHE_ESTIMATES = KVCacheBenchmark._MODEL_CACHE_ESTIMATES
        bm._MODEL_CACHE_DEFAULT = KVCacheBenchmark._MODEL_CACHE_DEFAULT
        bm.model = "llama3.1-8b"  # per_token=4096, seq=8192
        bm.num_users = 10

        # cache_per_user_mb = (4096 * 8192) / (1024*1024) = 32
        # total_cache_mb = 32 * 10 = 320
        # expected = int(320 * 1024 * 1024) = 335544320
        expected = int(((4096 * 8192) / (1024 * 1024)) * 10 * 1024 * 1024)
        result = KVCacheBenchmark.required_bytes_for_capacity_gate(bm)
        assert result == expected
        # 1x lock guard — make sure we did NOT multiply by 2.
        assert result != expected * 2

    def test_destination_is_cache_path(self):
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        bm = MagicMock(spec=KVCacheBenchmark)
        bm.cache_dir = "/nvme/kvcache"
        result = KVCacheBenchmark._capacity_gate_destination(bm)
        assert result == "/nvme/kvcache"

    def test_uses_kvcache_internal_model_table_not_config_py_constants(self):
        """A6 lock: required_bytes consults model_cache_estimates inline
        (per_token_bytes/typical_sequence), NOT LLAMA3_8B/etc from config.py.

        The contract is that for unknown models we fall back to the inline
        default (per_token=4096, seq=4096), NOT raise KeyError on a config.py
        constants lookup.
        """
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        bm = MagicMock(spec=KVCacheBenchmark)
        bm._MODEL_CACHE_ESTIMATES = KVCacheBenchmark._MODEL_CACHE_ESTIMATES
        bm._MODEL_CACHE_DEFAULT = KVCacheBenchmark._MODEL_CACHE_DEFAULT
        # An unknown model name MUST fall through to the inline default,
        # NOT raise (i.e., the impl does NOT route through LLM_SIZE_BY_RANK).
        bm.model = "totally-made-up-model"
        bm.num_users = 1
        # default per_token=4096, seq=4096 -> 16 MB total -> 16777216 bytes
        expected = int(((4096 * 4096) / (1024 * 1024)) * 1 * 1024 * 1024)
        result = KVCacheBenchmark.required_bytes_for_capacity_gate(bm)
        assert result == expected
