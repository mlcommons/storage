"""
Tests for KVCacheBenchmark class in mlpstorage.benchmarks.kvcache module.

Tests cover:
- MPI command generation with distributed execution
- Local execution without MPI wrapper
- Default num_processes behavior
- MPI flag passthrough (oversubscribe, allow-run-as-root, mpi-params)
- Cluster information collection for distributed runs
- _interruptible_sleep: what-if skip, chunked sleep, Ctrl-C propagation
- _aggregate_option_results: bandwidth sum, P95 max, partial failure, CPU-tier flag
- _write_validate_summary: output path, JSON schema, MLPSJsonEncoder usage
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from argparse import Namespace

from mlpstorage_py.config import BENCHMARK_TYPES, EXEC_TYPE


class TestKVCacheMPIExecution:
    """Tests for MPI execution support in KVCacheBenchmark."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for KV cache benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=None,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    @pytest.fixture
    def mock_benchmark(self, basic_args, tmp_path):
        """Create a mocked KVCacheBenchmark instance."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")
            return benchmark

    def test_local_execution_no_mpi_wrapper(self, basic_args, tmp_path):
        """Command should NOT have MPI wrapper when exec_type is None."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd
        assert 'mpiexec' not in cmd
        assert '--model llama3.1-8b' in cmd

    def test_docker_execution_no_mpi_wrapper(self, basic_args, tmp_path):
        """Command should NOT have MPI wrapper when exec_type is DOCKER."""
        basic_args.exec_type = EXEC_TYPE.DOCKER

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd
        assert 'mpiexec' not in cmd

    def test_mpi_execution_adds_wrapper(self, basic_args, tmp_path):
        """Command should have MPI wrapper when exec_type is MPI with hosts."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2']
        basic_args.num_processes = 4

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert cmd.startswith('mpirun')
        assert '-n 4' in cmd
        assert 'host1' in cmd
        assert 'host2' in cmd
        assert '--model llama3.1-8b' in cmd

    def test_mpi_execution_empty_hosts_no_wrapper(self, basic_args, tmp_path):
        """Command should NOT have MPI wrapper when hosts is empty list."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = []
        basic_args.num_processes = 4

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd

    def test_mpi_execution_defaults_num_processes_to_host_count(self, basic_args, tmp_path):
        """num_processes should default to len(hosts) when not specified."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2', 'host3']
        basic_args.num_processes = None  # Not specified

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert cmd.startswith('mpirun')
        assert '-n 3' in cmd  # 3 hosts = 3 processes

    def test_mpi_execution_oversubscribe_flag(self, basic_args, tmp_path):
        """MPI command should include --oversubscribe when flag is set."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1']
        basic_args.num_processes = 4
        basic_args.oversubscribe = True

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert '--oversubscribe' in cmd

    def test_mpi_execution_allow_run_as_root_flag(self, basic_args, tmp_path):
        """MPI command should include --allow-run-as-root when flag is set."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1']
        basic_args.num_processes = 4
        basic_args.allow_run_as_root = True

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert '--allow-run-as-root' in cmd

    def test_mpi_execution_uses_mpiexec(self, basic_args, tmp_path):
        """MPI command should use mpiexec when specified."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2']
        basic_args.num_processes = 4
        basic_args.mpi_bin = 'mpiexec'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert cmd.startswith('mpiexec')


class TestKVCacheClusterCollection:
    """Tests for cluster information collection in KVCacheBenchmark."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for KV cache benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=None,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    def test_cluster_collection_called_for_run_command(self, basic_args, tmp_path):
        """Should collect cluster information for run command."""
        basic_args.command = 'run'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = MagicMock()
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        mock_cluster.assert_called_once()
        assert hasattr(benchmark, 'cluster_information')

    def test_cluster_collection_not_called_for_datasize_command(self, basic_args, tmp_path):
        """Should NOT collect cluster information for datasize command."""
        basic_args.command = 'datasize'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        mock_cluster.assert_not_called()

    def test_cluster_collection_not_called_for_validate_command(self, tmp_path):
        """Should NOT collect cluster information for validate command (G1)."""
        bm = _make_validate_benchmark(tmp_path)
        from mlpstorage_py.interfaces import BenchmarkCommand
        assert BenchmarkCommand.VALIDATE in bm._get_supported_commands()

        # The two-patch pattern inside _make_validate_benchmark already verified
        # mock_cluster.assert_not_called() implicitly — the benchmark was constructed
        # without triggering cluster collection. Re-verify via a fresh construction:
        args = Namespace(
            debug=False, verbose=False, what_if=False, stream_log_level='INFO',
            results_dir=str(tmp_path), command='validate',
            npernode=1, seed=42, cache_dir='/tmp/kv', trials=1, inter_option_delay=0,
            kvcache_bin_path=None, config=None,
            hosts=['localhost'], mpi_bin='mpirun',
            oversubscribe=False, allow_run_as_root=False, mpi_params=None, mpi_btl='auto',
            model='llama3.1-8b', num_users=100, duration=60,
            gpu_mem_gb=0, cpu_mem_gb=0,
            generation_mode='none', performance_profile='latency',
            num_processes=None, exec_type=None,
            closed=False, open=False,
        )
        output_dir = str(tmp_path / 'validate_g1')
        os.makedirs(output_dir, exist_ok=True)
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            mock_gen.return_value = output_dir
            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            KVCacheBenchmark(args, run_datetime='20260523_120000')
        mock_cluster.assert_not_called()


class TestKVCacheNumProcessesStorage:
    """Tests for num_processes storage in KVCacheBenchmark."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for KV cache benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=8,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    def test_num_processes_stored_from_args(self, basic_args, tmp_path):
        """Should store num_processes from args."""
        basic_args.num_processes = 16

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes == 16

    def test_num_processes_none_when_not_provided(self, basic_args, tmp_path):
        """Should be None when num_processes not in args."""
        del basic_args.num_processes  # Remove attribute

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes is None


class TestKVCacheMetadata:
    """Test metadata structure for history integration."""

    @pytest.fixture
    def base_args(self, tmp_path):
        """Create base args for KV cache benchmark metadata tests."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=None,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = MagicMock()
        logger.status = MagicMock()
        logger.info = MagicMock()
        logger.debug = MagicMock()
        logger.warning = MagicMock()
        logger.verboser = MagicMock()
        logger.verbose = MagicMock()
        return logger

    def test_metadata_has_required_fields(self, base_args, mock_logger, tmp_path):
        """Verify metadata includes fields required by history module."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        # Required by history module
        assert 'benchmark_type' in meta
        assert 'model' in meta
        assert 'command' in meta
        assert 'run_datetime' in meta
        assert 'result_dir' in meta

    def test_metadata_includes_kvcache_specific_fields(self, base_args, mock_logger, tmp_path):
        """Verify KV cache specific metadata fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert 'kvcache_model' in meta
        assert 'num_users' in meta
        assert 'duration' in meta
        assert 'gpu_mem_gb' in meta
        assert 'cpu_mem_gb' in meta
        assert 'generation_mode' in meta
        assert 'performance_profile' in meta

    def test_metadata_includes_distributed_info(self, base_args, mock_logger, tmp_path):
        """Verify metadata includes distributed execution info."""
        base_args.exec_type = EXEC_TYPE.MPI
        base_args.hosts = ['host1', 'host2']
        base_args.num_processes = 4

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert 'num_processes' in meta
        assert meta['num_processes'] == 4
        assert 'hosts' in meta
        assert meta['hosts'] == ['host1', 'host2']
        assert 'exec_type' in meta

    def test_metadata_model_consistency(self, base_args, mock_logger, tmp_path):
        """Verify 'model' field matches 'kvcache_model' for history compatibility."""
        base_args.model = 'llama3.1-70b-instruct'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert meta['model'] == 'llama3.1-70b-instruct'
        assert meta['kvcache_model'] == 'llama3.1-70b-instruct'

    def test_metadata_without_distributed_info(self, base_args, mock_logger, tmp_path):
        """Verify metadata works correctly without distributed execution info."""
        # exec_type, hosts, num_processes are None by default in base_args

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        # num_processes should be included but can be None
        assert 'num_processes' in meta
        assert meta['num_processes'] is None
        # hosts and exec_type should not be in metadata when not set
        assert 'hosts' not in meta
        assert 'exec_type' not in meta


# ---------------------------------------------------------------------------
# Helper fixture shared by AGG tests
# ---------------------------------------------------------------------------

def _make_validate_benchmark(tmp_path, what_if=False):
    """Instantiate KVCacheBenchmark for command='validate' with mocked deps."""
    args = Namespace(
        debug=False,
        verbose=False,
        what_if=what_if,
        stream_log_level='INFO',
        results_dir=str(tmp_path),
        command='validate',
        npernode=2,
        seed=42,
        cache_dir='/tmp/kv',
        trials=3,
        inter_option_delay=20,
        kvcache_bin_path=None,
        config=None,
        hosts=['localhost'],
        mpi_bin='mpirun',
        oversubscribe=False,
        allow_run_as_root=False,
        mpi_params=None,
        mpi_btl='auto',
        model='llama3.1-8b',
        num_users=100,
        duration=60,
        gpu_mem_gb=0,
        cpu_mem_gb=0,
        generation_mode='none',
        performance_profile='latency',
        num_processes=None,
        exec_type=None,
        closed=False,
        open=False,
    )
    output_dir = str(tmp_path / 'validate_output')
    os.makedirs(output_dir, exist_ok=True)
    with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
         patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information',
               return_value=None):
        mock_gen.return_value = output_dir
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
        bm = KVCacheBenchmark(args, run_datetime='20260523_120000')
    return bm


class TestInterruptibleSleep:
    """Tests for KVCacheBenchmark._interruptible_sleep."""

    def test_returns_immediately_with_zero_seconds(self, tmp_path):
        """Sleep of 0 seconds returns immediately."""
        bm = _make_validate_benchmark(tmp_path)
        # Must not raise, must complete quickly
        bm._interruptible_sleep(0)

    def test_skips_sleep_in_what_if_mode(self, tmp_path):
        """_interruptible_sleep must return immediately when what_if=True."""
        bm = _make_validate_benchmark(tmp_path, what_if=True)
        with patch('time.sleep') as mock_sleep:
            bm._interruptible_sleep(20)
        mock_sleep.assert_not_called()

    def test_calls_time_sleep_in_1s_chunks(self, tmp_path):
        """_interruptible_sleep(3) calls time.sleep(1) three times."""
        bm = _make_validate_benchmark(tmp_path, what_if=False)
        with patch('time.sleep') as mock_sleep:
            bm._interruptible_sleep(3)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(1)

    def test_propagates_keyboard_interrupt(self, tmp_path):
        """KeyboardInterrupt raised inside sleep should propagate out."""
        bm = _make_validate_benchmark(tmp_path, what_if=False)
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                bm._interruptible_sleep(5)


class TestAggregateOptionResults:
    """Tests for KVCacheBenchmark._aggregate_option_results."""

    def _make_rank_file(self, rank_dir, bw, p95, storage_entries=100):
        """Write a synthetic rank output JSON file."""
        rank_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'summary': {
                'cache_stats': {
                    'tier_storage_read_bandwidth_gbps': bw,
                    'storage_entries': storage_entries,
                },
                'storage_io_latency_ms': {'p95': p95},
            }
        }
        (rank_dir / 'kvcache_results_20260523_120000.json').write_text(json.dumps(data))

    def test_sums_bandwidth_across_ranks(self, tmp_path):
        """aggregated_bandwidth_gbps == sum of all rank values."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.5, p95=10.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.5, p95=15.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['aggregated_bandwidth_gbps'] == pytest.approx(4.0)

    def test_takes_max_p95_latency_across_ranks(self, tmp_path):
        """aggregated_p95_latency_ms == max of all rank p95 values."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.5, p95=10.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.5, p95=15.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['aggregated_p95_latency_ms'] == pytest.approx(15.0)

    def test_no_partial_failure_when_all_files_present(self, tmp_path):
        """partial_failure is False when all rank files exist."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=5.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.0, p95=8.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['partial_failure'] is False
        assert result['missing_files'] == []

    def test_partial_failure_when_rank_file_missing(self, tmp_path):
        """partial_failure is True when a rank directory has no result file."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        # Only rank_0 present; rank_1 is missing
        self._make_rank_file(trial_dir / 'rank_0', bw=2.0, p95=8.0)
        # rank_1 directory exists but has no json file
        (trial_dir / 'rank_1').mkdir(parents=True, exist_ok=True)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['partial_failure'] is True
        assert len(result['missing_files']) == 1

    def test_cpu_tier_ranks_populated_when_storage_entries_zero(self, tmp_path):
        """cpu_tier_ranks is populated and bandwidth included when storage_entries==0."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=0.0, p95=5.0, storage_entries=0)
        self._make_rank_file(trial_dir / 'rank_1', bw=0.0, p95=5.0, storage_entries=0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        # AGG-04: 0 bandwidth is included, not a failure
        assert result['aggregated_bandwidth_gbps'] == pytest.approx(0.0)
        assert result['partial_failure'] is False
        assert len(result['cpu_tier_ranks']) == 2

    def test_cpu_tier_log_message_contains_required_text(self, tmp_path):
        """Logger must log 'working set served from CPU tier' for storage_entries==0."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=0.0, p95=5.0, storage_entries=0)

        log_messages = []
        original_info = bm.logger.info
        def capture_info(msg, *a, **kw):
            log_messages.append(str(msg))
            if callable(original_info):
                try:
                    original_info(msg, *a, **kw)
                except Exception:
                    pass
        bm.logger.info = capture_info

        bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        assert any('working set served from CPU tier' in m for m in log_messages), \
            f"Expected 'working set served from CPU tier' in log messages, got: {log_messages}"

    def test_result_structure_has_required_keys(self, tmp_path):
        """Return dict must contain all AGG-06 required keys."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=5.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        required_keys = {
            'option', 'aggregated_bandwidth_gbps', 'aggregated_p95_latency_ms',
            'rank_count', 'trial_count', 'partial_failure', 'missing_files', 'cpu_tier_ranks',
        }
        assert required_keys.issubset(set(result.keys()))

    def test_aggregates_across_multiple_trials(self, tmp_path):
        """Aggregation spans multiple trial directories."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dirs = []
        for t in range(2):
            trial_dir = tmp_path / f'trial_{t}'
            self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=10.0)
            trial_dirs.append(str(trial_dir))

        result = bm._aggregate_option_results(1, trial_dirs, expected_rank_count=1)

        # 2 trials × 1 rank × 1.0 GBps = 2.0
        assert result['aggregated_bandwidth_gbps'] == pytest.approx(2.0)
        assert result['trial_count'] == 2

    def test_uses_glob_not_constructed_filename(self, tmp_path):
        """Discovery must use glob so clock-drift timestamps are tolerated."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        rank_dir = trial_dir / 'rank_0'
        rank_dir.mkdir(parents=True)
        # Write file with a different timestamp than run_datetime
        data = {
            'summary': {
                'cache_stats': {'tier_storage_read_bandwidth_gbps': 3.0, 'storage_entries': 50},
                'storage_io_latency_ms': {'p95': 7.0},
            }
        }
        (rank_dir / 'kvcache_results_20260523_130055.json').write_text(json.dumps(data))

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        assert result['aggregated_bandwidth_gbps'] == pytest.approx(3.0)
        assert result['partial_failure'] is False

    def test_none_p95_when_no_successful_reads(self, tmp_path):
        """aggregated_p95_latency_ms is None when all rank files are missing."""
        bm = _make_validate_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        # Both rank dirs exist but have no json files
        (trial_dir / 'rank_0').mkdir(parents=True)
        (trial_dir / 'rank_1').mkdir(parents=True)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['aggregated_p95_latency_ms'] is None
        assert result['partial_failure'] is True


class TestWriteValidateSummary:
    """Tests for KVCacheBenchmark._write_validate_summary."""

    def _option_result(self, option=1, bw=3.0, p95=12.0, partial=False):
        return {
            'option': option,
            'aggregated_bandwidth_gbps': bw,
            'aggregated_p95_latency_ms': p95,
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': partial,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }

    def test_writes_file_to_run_result_output(self, tmp_path):
        """Summary JSON is written to run_result_output with correct filename."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result()}
        bm._write_validate_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        expected = Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json'
        assert expected.exists(), f"Expected summary at {expected}"

    def test_schema_version_is_1_0(self, tmp_path):
        """Written JSON must have schema_version='1.0'."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        bm._write_validate_summary({1: self._option_result()}, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['schema_version'] == '1.0'

    def test_summary_includes_required_keys(self, tmp_path):
        """JSON must contain all AGG-06 top-level keys."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        bm._write_validate_summary({1: self._option_result()}, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json') as f:
            data = json.load(f)

        required = {'schema_version', 'run_datetime', 'npernode', 'host_count',
                    'total_ranks', 'trials_per_option', 'options', 'partial_failure'}
        assert required.issubset(set(data.keys()))

    def test_partial_failure_true_when_any_option_fails(self, tmp_path):
        """Top-level partial_failure is True when any option has partial_failure=True."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {
            1: self._option_result(partial=False),
            2: self._option_result(option=2, partial=True),
            3: self._option_result(option=3, partial=False),
        }
        bm._write_validate_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['partial_failure'] is True

    def test_partial_failure_false_when_no_option_fails(self, tmp_path):
        """Top-level partial_failure is False when no option has partial_failure=True."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result(partial=False)}
        bm._write_validate_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['partial_failure'] is False

    def test_options_key_contains_per_option_data(self, tmp_path):
        """The 'options' key must hold the per-option result dict."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result(bw=5.0)}
        bm._write_validate_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_validate_summary_20260523_120000.json') as f:
            data = json.load(f)
        # JSON keys are strings after serialization
        options = data['options']
        assert '1' in options or 1 in options

    def test_write_validate_summary_does_not_raise_with_float_values(self, tmp_path):
        """MLPSJsonEncoder must serialize standard Python floats without raising (G5)."""
        bm = _make_validate_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_g5')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_result = {
            'option': 1,
            'aggregated_bandwidth_gbps': float(3.5),
            'aggregated_p95_latency_ms': float(12.0),
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': False,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }
        bm._write_validate_summary({1: option_result}, npernode=2, host_count=1, total_ranks=2, trials=3)
        summary_files = list(Path(output_dir).glob('kvcache_validate_summary_*.json'))
        assert len(summary_files) == 1


class TestExecuteValidate:
    """Tests for KVCacheBenchmark._execute_validate and command_method_map wiring.

    Covers:
    - command_method_map contains 'validate' key mapping to _execute_validate (DIST-01)
    - __init__ cluster guard unchanged — not extended to 'validate'
    - _execute_validate returns 0
    - _execute_command called 3 times per run (once per option) with trials=1 (DIST-02, DIST-04)
    - mpirun command contains '--mca orte_abort_on_non_zero_status 0' (DIST-08)
    - mpirun command contains '--npernode N' (DIST-03)
    - wrapper receives --option, --seed, --base-output-dir, --cache-dir (DIST-07)
    - per-option/trial dirs created with correct naming (option_{N}/trial_{T}/)
    - _interruptible_sleep called 2 times (after options 1 and 2; not after 3) (DIST-05)
    - _aggregate_option_results called 3 times when what_if=False
    - _aggregate_option_results NOT called when what_if=True (DIST-06)
    - _write_validate_summary called once when what_if=False
    - _write_validate_summary NOT called when what_if=True
    - write_metadata called regardless of what_if
    """

    @pytest.fixture
    def bm(self, tmp_path):
        """Benchmark instance for validate command."""
        return _make_validate_benchmark(tmp_path, what_if=False)

    @pytest.fixture
    def bm_whatif(self, tmp_path):
        """Benchmark instance for validate --what-if."""
        return _make_validate_benchmark(tmp_path, what_if=True)

    @pytest.fixture
    def fake_agg_result(self):
        """Return value for _aggregate_option_results mock."""
        return {
            'option': 1,
            'aggregated_bandwidth_gbps': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': False,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }

    def test_validate_in_command_method_map(self, bm):
        """'validate' key must exist in command_method_map and map to _execute_validate."""
        assert 'validate' in bm.command_method_map
        assert bm.command_method_map['validate'] == bm._execute_validate

    def test_cluster_guard_unchanged(self, tmp_path):
        """__init__ must NOT collect cluster info for command='validate'."""
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch.object(KVCacheBenchmark, '_collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / 'validate_cluster_guard')
            os.makedirs(output_dir, exist_ok=True)
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            args = Namespace(
                debug=False, verbose=False, what_if=False,
                stream_log_level='INFO',
                results_dir=str(tmp_path),
                command='validate',
                npernode=2, seed=42, cache_dir='/tmp/kv', trials=1,
                inter_option_delay=0, kvcache_bin_path=None, config=None,
                hosts=['localhost'], mpi_bin='mpirun',
                oversubscribe=False, allow_run_as_root=False,
                mpi_params=None, mpi_btl='auto',
                model='llama3.1-8b', num_users=100, duration=60,
                gpu_mem_gb=0, cpu_mem_gb=0,
                generation_mode='none', performance_profile='latency',
                num_processes=None, exec_type=None,
                closed=False, open=False,
            )
            KVCacheBenchmark(args, run_datetime='20260523_120000')
        mock_cluster.assert_not_called()

    def test_execute_validate_returns_zero(self, bm, fake_agg_result):
        """_execute_validate must return 0 on success."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_validate()
        assert rc == 0

    def test_execute_command_called_3_times_for_3_options(self, bm, fake_agg_result):
        """_execute_command must be called once per option (3x) with trials=1."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert len(executed_cmds) == 3, f"Expected 3 _execute_command calls, got {len(executed_cmds)}"

    def test_mpirun_contains_mca_orte_flag(self, bm, fake_agg_result):
        """mpirun command must contain '--mca orte_abort_on_non_zero_status 0' (DIST-08)."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert executed_cmds, "No commands were executed"
        cmd0 = executed_cmds[0]
        assert '--mca orte_abort_on_non_zero_status 0' in cmd0, \
            f"Missing --mca orte_abort_on_non_zero_status 0 in: {cmd0}"

    def test_mpirun_contains_npernode(self, bm, fake_agg_result):
        """mpirun command must contain '--npernode 2' when npernode=2 (DIST-03)."""
        bm.args.npernode = 2
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert '--npernode 2' in executed_cmds[0], \
            f"Missing --npernode 2 in: {executed_cmds[0]}"

    def test_wrapper_receives_option_seed_output_cache(self, bm, fake_agg_result):
        """Wrapper command must include --option, --seed, --base-output-dir, --cache-dir (DIST-07)."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.seed = 42
        bm.args.cache_dir = '/tmp/kv'
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        cmd0 = executed_cmds[0]
        assert '--option 1' in cmd0, f"Missing --option 1 in: {cmd0}"
        assert '--seed 42' in cmd0, f"Missing --seed 42 in: {cmd0}"
        assert '--base-output-dir' in cmd0, f"Missing --base-output-dir in: {cmd0}"
        assert '--cache-dir /tmp/kv' in cmd0, f"Missing --cache-dir in: {cmd0}"

    def test_per_option_trial_dirs_created(self, bm, fake_agg_result, tmp_path):
        """option_{N}/trial_{T}/ directories must be created."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        # At least one option/trial dir must exist beneath run_result_output
        run_out = Path(bm.run_result_output)
        option1_trial0 = run_out / 'option_1' / 'trial_0'
        assert option1_trial0.exists(), f"Expected {option1_trial0} to exist"

    def test_option_trial_dirs_in_command_path(self, bm, fake_agg_result):
        """Command must reference option_N/trial_T subdirectory in --base-output-dir."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        cmd0 = executed_cmds[0]
        assert 'option_1' in cmd0, f"Missing option_1 in: {cmd0}"
        assert 'trial_0' in cmd0, f"Missing trial_0 in: {cmd0}"

    def test_interruptible_sleep_called_2_times_not_3(self, bm, fake_agg_result):
        """_interruptible_sleep called after options 1 and 2 but NOT after option 3."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 5
        sleep_calls = []
        def fake_sleep(seconds):
            sleep_calls.append(seconds)
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep', side_effect=fake_sleep), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert len(sleep_calls) == 2, f"Expected 2 sleep calls, got {len(sleep_calls)}"
        assert all(s == 5 for s in sleep_calls), f"Expected delay=5 for all, got {sleep_calls}"

    def test_aggregate_called_3_times_when_not_what_if(self, bm, fake_agg_result):
        """_aggregate_option_results called once per option (3x) when what_if=False."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result) as mock_agg, \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert mock_agg.call_count == 3, f"Expected 3 aggregate calls, got {mock_agg.call_count}"

    def test_aggregate_not_called_when_what_if(self, bm_whatif):
        """_aggregate_option_results must NOT be called when what_if=True."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results') as mock_agg, \
             patch.object(bm_whatif, '_write_validate_summary') as mock_ws, \
             patch.object(bm_whatif, 'write_metadata'):
            bm_whatif._execute_validate()
        assert mock_agg.call_count == 0, f"Expected 0 aggregate calls in what-if, got {mock_agg.call_count}"

    def test_write_summary_called_once_when_not_what_if(self, bm, fake_agg_result):
        """_write_validate_summary called once after all options when what_if=False."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary') as mock_ws, \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert mock_ws.call_count == 1, f"Expected 1 summary write, got {mock_ws.call_count}"

    def test_write_summary_not_called_when_what_if(self, bm_whatif):
        """_write_validate_summary must NOT be called when what_if=True."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results'), \
             patch.object(bm_whatif, '_write_validate_summary') as mock_ws, \
             patch.object(bm_whatif, 'write_metadata'):
            bm_whatif._execute_validate()
        assert mock_ws.call_count == 0, f"Expected 0 summary writes in what-if, got {mock_ws.call_count}"

    def test_write_metadata_called_regardless_of_what_if(self, bm_whatif):
        """write_metadata must be called even in what-if mode."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results'), \
             patch.object(bm_whatif, '_write_validate_summary'), \
             patch.object(bm_whatif, 'write_metadata') as mock_meta:
            bm_whatif._execute_validate()
        assert mock_meta.call_count == 1, "write_metadata must be called even in what-if"

    def test_multiple_trials_per_option(self, bm, fake_agg_result):
        """With trials=3, _execute_command called 9 times (3 options × 3 trials)."""
        bm.args.trials = 3
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        assert len(executed_cmds) == 9, f"Expected 9 commands (3 options × 3 trials), got {len(executed_cmds)}"

    def test_execute_command_targets_mlperf_wrapper_not_kvcache(self, bm, fake_agg_result):
        """Command must reference mlperf_wrapper.py, not kv-cache.py (G2)."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        cmd0 = executed_cmds[0]
        assert 'mlperf_wrapper.py' in cmd0, f"Expected mlperf_wrapper.py in cmd, got: {cmd0}"
        assert 'kv-cache.py' not in cmd0, f"Must not reference kv-cache.py directly: {cmd0}"

    def test_aggregate_receives_correct_trial_dirs_for_two_trials(self, bm, fake_agg_result):
        """_aggregate_option_results must receive trial_dirs for all trials (G3)."""
        bm.args.trials = 2
        bm.args.inter_option_delay = 0
        agg_calls = []
        def fake_agg(option, trial_dirs, expected_rank_count):
            agg_calls.append((option, list(trial_dirs)))
            return fake_agg_result
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', side_effect=fake_agg), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_validate()
        # agg_calls[0] is for option 1
        option_1_dirs = agg_calls[0][1]
        assert len(option_1_dirs) == 2
        assert any('trial_0' in str(d) for d in option_1_dirs)
        assert any('trial_1' in str(d) for d in option_1_dirs)

    def test_localhost_fallback_when_hosts_is_none(self, bm, fake_agg_result):
        """When hosts is None, _execute_validate must complete without raising (G4)."""
        bm.args.hosts = None
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_validate_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_validate()
        assert rc == 0
