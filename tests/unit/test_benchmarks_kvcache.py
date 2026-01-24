"""
Tests for KVCacheBenchmark class in mlpstorage.benchmarks.kvcache module.

Tests cover:
- MPI command generation with distributed execution
- Local execution without MPI wrapper
- Default num_processes behavior
- MPI flag passthrough (oversubscribe, allow-run-as-root, mpi-params)
- Cluster information collection for distributed runs
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from argparse import Namespace

from mlpstorage.config import BENCHMARK_TYPES, EXEC_TYPE


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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")
            return benchmark

    def test_local_execution_no_mpi_wrapper(self, basic_args, tmp_path):
        """Command should NOT have MPI wrapper when exec_type is None."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd
        assert 'mpiexec' not in cmd
        assert '--model llama3.1-8b' in cmd

    def test_docker_execution_no_mpi_wrapper(self, basic_args, tmp_path):
        """Command should NOT have MPI wrapper when exec_type is DOCKER."""
        basic_args.exec_type = EXEC_TYPE.DOCKER

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd
        assert 'mpiexec' not in cmd

    def test_mpi_execution_adds_wrapper(self, basic_args, tmp_path):
        """Command should have MPI wrapper when exec_type is MPI with hosts."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2']
        basic_args.num_processes = 4

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
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

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert 'mpirun' not in cmd

    def test_mpi_execution_defaults_num_processes_to_host_count(self, basic_args, tmp_path):
        """num_processes should default to len(hosts) when not specified."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2', 'host3']
        basic_args.num_processes = None  # Not specified

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
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

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert '--oversubscribe' in cmd

    def test_mpi_execution_allow_run_as_root_flag(self, basic_args, tmp_path):
        """MPI command should include --allow-run-as-root when flag is set."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1']
        basic_args.num_processes = 4
        basic_args.allow_run_as_root = True

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

            cmd = benchmark._build_kvcache_command()

        assert '--allow-run-as-root' in cmd

    def test_mpi_execution_uses_mpiexec(self, basic_args, tmp_path):
        """MPI command should use mpiexec when specified."""
        basic_args.exec_type = EXEC_TYPE.MPI
        basic_args.hosts = ['host1', 'host2']
        basic_args.num_processes = 4
        basic_args.mpi_bin = 'mpiexec'

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
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

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = MagicMock()
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        mock_cluster.assert_called_once()
        assert hasattr(benchmark, 'cluster_information')

    def test_cluster_collection_not_called_for_datasize_command(self, basic_args, tmp_path):
        """Should NOT collect cluster information for datasize command."""
        basic_args.command = 'datasize'

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

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

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes == 16

    def test_num_processes_none_when_not_provided(self, basic_args, tmp_path):
        """Should be None when num_processes not in args."""
        del basic_args.num_processes  # Remove attribute

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes is None
