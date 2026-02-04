#!/usr/bin/env python3
"""
Tests for mlpstorage.benchmarks.base module.

This module tests the base Benchmark class, specifically the cluster
information collection methods added in Phase 3.

Run with:
    pytest mlpstorage/tests/test_benchmarks.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from mlpstorage.config import EXEC_TYPE, BENCHMARK_TYPES


class MockLogger:
    """Mock logger for testing."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def verbose(self, msg): pass
    def verboser(self, msg): pass
    def status(self, msg): pass


@pytest.fixture
def mock_logger():
    """Return a mock logger."""
    return MockLogger()


@pytest.fixture
def base_args():
    """Create basic args namespace for testing."""
    return SimpleNamespace(
        hosts=['host1', 'host2'],
        command='run',
        exec_type=EXEC_TYPE.MPI,
        mpi_bin='mpirun',
        allow_run_as_root=False,
        debug=False,
        verbose=False,
        stream_log_level='INFO',
        results_dir='/tmp/results',
        what_if=False,
    )


# =============================================================================
# Tests for _should_collect_cluster_info
# =============================================================================

class TestShouldCollectClusterInfo:
    """Tests for Benchmark._should_collect_cluster_info method."""

    def test_returns_true_with_hosts_and_run_command(self, base_args, mock_logger):
        """Should return True when hosts are specified and command is 'run'."""
        from mlpstorage.benchmarks.base import Benchmark

        # Create a concrete subclass for testing
        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            assert benchmark._should_collect_cluster_info() is True

    def test_returns_false_without_hosts(self, base_args, mock_logger):
        """Should return False when hosts are not specified."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        base_args.hosts = []

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            assert benchmark._should_collect_cluster_info() is False

    def test_returns_false_for_datagen_command(self, base_args, mock_logger):
        """Should return False when command is 'datagen'."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        base_args.command = 'datagen'

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            assert benchmark._should_collect_cluster_info() is False

    def test_returns_false_for_configview_command(self, base_args, mock_logger):
        """Should return False when command is 'configview'."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        base_args.command = 'configview'

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            assert benchmark._should_collect_cluster_info() is False

    def test_returns_false_when_skip_cluster_collection_set(self, base_args, mock_logger):
        """Should return False when skip_cluster_collection is True."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        base_args.skip_cluster_collection = True

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            assert benchmark._should_collect_cluster_info() is False


# =============================================================================
# Tests for _collect_cluster_information
# =============================================================================

class TestCollectClusterInformation:
    """Tests for Benchmark._collect_cluster_information method."""

    def test_returns_none_when_should_not_collect(self, base_args, mock_logger):
        """Should return None when _should_collect_cluster_info returns False."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        base_args.hosts = []  # No hosts = should not collect

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            result = benchmark._collect_cluster_information()
            assert result is None

    def test_returns_none_when_not_mpi_exec_type(self, base_args, mock_logger):
        """Should return None when exec_type is not MPI."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        # Use DOCKER instead of NONE (EXEC_TYPE only has MPI and DOCKER)
        base_args.exec_type = EXEC_TYPE.DOCKER

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            result = benchmark._collect_cluster_information()
            assert result is None

    def test_calls_collect_cluster_info_with_correct_params(self, base_args, mock_logger):
        """Should call collect_cluster_info with correct parameters."""
        from mlpstorage.benchmarks.base import Benchmark
        from mlpstorage.rules import ClusterInformation

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        mock_collected_data = {
            'host1': {
                'hostname': 'host1',
                'meminfo': {'MemTotal': 16384000},
            },
            '_metadata': {
                'collection_method': 'mpi',
                'collection_timestamp': '2024-01-01T00:00:00Z',
            }
        }

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            with patch('mlpstorage.benchmarks.base.collect_cluster_info') as mock_collect:
                mock_collect.return_value = mock_collected_data

                result = benchmark._collect_cluster_information()

                # Verify collect_cluster_info was called with correct args
                mock_collect.assert_called_once_with(
                    hosts=['host1', 'host2'],
                    mpi_bin='mpirun',
                    logger=mock_logger,
                    allow_run_as_root=False,
                    timeout_seconds=60,
                    fallback_to_local=True
                )

                # Verify result is a ClusterInformation instance
                assert isinstance(result, ClusterInformation)
                assert result.collection_method == 'mpi'

    def test_returns_none_on_exception(self, base_args, mock_logger):
        """Should return None and log warning when collection fails."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger

            with patch('mlpstorage.benchmarks.base.collect_cluster_info') as mock_collect:
                mock_collect.side_effect = Exception("MPI failed")

                result = benchmark._collect_cluster_information()

                assert result is None


# =============================================================================
# Tests for DLIOBenchmark.accumulate_host_info
# =============================================================================

class TestDLIOBenchmarkAccumulateHostInfo:
    """Tests for DLIOBenchmark.accumulate_host_info method."""

    def test_uses_mpi_collection_when_available(self, mock_logger):
        """Should use MPI collection when it succeeds."""
        from mlpstorage.benchmarks.dlio import TrainingBenchmark
        from mlpstorage.rules import ClusterInformation

        args = SimpleNamespace(
            hosts=['host1', 'host2'],
            command='run',
            exec_type=EXEC_TYPE.MPI,
            mpi_bin='mpirun',
            allow_run_as_root=False,
            client_host_memory_in_gb=64,
            debug=False,
            verbose=False,
        )

        mock_cluster_info = MagicMock(spec=ClusterInformation)
        mock_cluster_info.num_hosts = 2
        mock_cluster_info.total_memory_bytes = 128 * 1024**3

        with patch.object(TrainingBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TrainingBenchmark.__new__(TrainingBenchmark)
            benchmark.args = args
            benchmark.logger = mock_logger

            with patch.object(benchmark, '_collect_cluster_information') as mock_collect:
                mock_collect.return_value = mock_cluster_info

                result = benchmark.accumulate_host_info(args)

                mock_collect.assert_called_once()
                assert result == mock_cluster_info

    def test_falls_back_to_args_when_mpi_fails(self, mock_logger):
        """Should fall back to CLI args when MPI collection returns None."""
        from mlpstorage.benchmarks.dlio import TrainingBenchmark
        from mlpstorage.rules import ClusterInformation

        args = SimpleNamespace(
            hosts=['host1', 'host2'],
            command='run',
            exec_type=EXEC_TYPE.MPI,
            mpi_bin='mpirun',
            allow_run_as_root=False,
            client_host_memory_in_gb=64,
            debug=False,
            verbose=False,
        )

        with patch.object(TrainingBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TrainingBenchmark.__new__(TrainingBenchmark)
            benchmark.args = args
            benchmark.logger = mock_logger

            with patch.object(benchmark, '_collect_cluster_information') as mock_collect:
                mock_collect.return_value = None  # MPI collection failed

                result = benchmark.accumulate_host_info(args)

                # Should have created ClusterInformation from args
                assert isinstance(result, ClusterInformation)
                assert result.collection_method == "args"
                assert result.num_hosts == 2
                # 64 GB per host * 2 hosts = 128 GB
                assert result.total_memory_bytes == 64 * 1024**3 * 2


# =============================================================================
# Tests for write_cluster_info
# =============================================================================

class TestWriteClusterInfo:
    """Tests for Benchmark.write_cluster_info method."""

    def test_writes_cluster_info_file(self, base_args, mock_logger, tmp_path):
        """Should write cluster info to JSON file."""
        from mlpstorage.benchmarks.base import Benchmark
        from mlpstorage.rules import ClusterInformation, HostInfo, HostMemoryInfo
        import json

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        host_info_list = [
            HostInfo(
                hostname='host1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
            ),
        ]
        cluster_info = ClusterInformation(host_info_list, mock_logger)

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger
            benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
            benchmark.run_result_output = str(tmp_path)
            benchmark.cluster_information = cluster_info

            benchmark.write_cluster_info()

            # Check file was created
            cluster_info_file = tmp_path / "training_cluster_info.json"
            assert cluster_info_file.exists()

            # Check file contents
            with open(cluster_info_file) as f:
                data = json.load(f)

            assert 'total_memory_bytes' in data
            assert 'num_hosts' in data

    def test_does_nothing_without_cluster_info(self, base_args, mock_logger, tmp_path):
        """Should do nothing if cluster_information is not set."""
        from mlpstorage.benchmarks.base import Benchmark

        class TestBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training
            def _run(self):
                pass

        with patch.object(TestBenchmark, '__init__', lambda x, *args, **kwargs: None):
            benchmark = TestBenchmark.__new__(TestBenchmark)
            benchmark.args = base_args
            benchmark.logger = mock_logger
            benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
            benchmark.run_result_output = str(tmp_path)
            # No cluster_information set

            benchmark.write_cluster_info()

            # No file should be created
            cluster_info_file = tmp_path / "training_cluster_info.json"
            assert not cluster_info_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
