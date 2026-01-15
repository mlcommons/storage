#!/usr/bin/env python3
"""
Tests for mlpstorage.rules module, specifically for ClusterInformation
and system_info metadata handling.

Run with:
    pytest mlpstorage/tests/test_rules.py -v
"""

import pytest
import logging
from unittest.mock import MagicMock, patch

from mlpstorage.rules import ClusterInformation, BenchmarkRun, BenchmarkResult


class MockLogger:
    """Mock logger for testing."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def verbose(self, msg): pass
    def verboser(self, msg): pass
    def ridiculous(self, msg): pass


@pytest.fixture
def mock_logger():
    """Return a mock logger."""
    return MockLogger()


class TestClusterInformationInfo:
    """Tests for ClusterInformation.info property."""

    def test_info_property_returns_dict(self, mock_logger):
        """info property should return a dictionary."""
        inst = ClusterInformation([], mock_logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = 1024 * 1024 * 1024  # 1GB
        inst.total_cores = 8

        result = inst.info
        assert isinstance(result, dict)

    def test_info_property_contains_total_memory_bytes(self, mock_logger):
        """info property should contain total_memory_bytes."""
        inst = ClusterInformation([], mock_logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = 1024 * 1024 * 1024  # 1GB
        inst.total_cores = 8

        result = inst.info
        assert 'total_memory_bytes' in result
        assert result['total_memory_bytes'] == 1024 * 1024 * 1024

    def test_info_property_contains_total_cores(self, mock_logger):
        """info property should contain total_cores."""
        inst = ClusterInformation([], mock_logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = 1024 * 1024 * 1024  # 1GB
        inst.total_cores = 8

        result = inst.info
        assert 'total_cores' in result
        assert result['total_cores'] == 8

    def test_info_matches_as_dict(self, mock_logger):
        """info property should return the same as as_dict()."""
        inst = ClusterInformation([], mock_logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = 2 * 1024 * 1024 * 1024  # 2GB
        inst.total_cores = 16

        assert inst.info == inst.as_dict()


class TestClusterInformationFromDict:
    """Tests for ClusterInformation.from_dict classmethod."""

    def test_from_dict_returns_instance(self, mock_logger):
        """from_dict should return a ClusterInformation instance."""
        data = {
            'total_memory_bytes': 1024 * 1024 * 1024,
            'total_cores': 8
        }
        result = ClusterInformation.from_dict(data, mock_logger)
        assert isinstance(result, ClusterInformation)

    def test_from_dict_sets_total_memory_bytes(self, mock_logger):
        """from_dict should correctly set total_memory_bytes."""
        data = {
            'total_memory_bytes': 2 * 1024 * 1024 * 1024,  # 2GB
            'total_cores': 8
        }
        result = ClusterInformation.from_dict(data, mock_logger)
        assert result.total_memory_bytes == 2 * 1024 * 1024 * 1024

    def test_from_dict_sets_total_cores(self, mock_logger):
        """from_dict should correctly set total_cores."""
        data = {
            'total_memory_bytes': 1024 * 1024 * 1024,
            'total_cores': 16
        }
        result = ClusterInformation.from_dict(data, mock_logger)
        assert result.total_cores == 16

    def test_from_dict_returns_none_for_none_data(self, mock_logger):
        """from_dict should return None if data is None."""
        result = ClusterInformation.from_dict(None, mock_logger)
        assert result is None

    def test_from_dict_returns_none_for_missing_total_memory_bytes(self, mock_logger):
        """from_dict should return None if total_memory_bytes is missing."""
        data = {'total_cores': 8}
        result = ClusterInformation.from_dict(data, mock_logger)
        assert result is None

    def test_from_dict_defaults_total_cores_to_zero(self, mock_logger):
        """from_dict should default total_cores to 0 if missing."""
        data = {'total_memory_bytes': 1024 * 1024 * 1024}
        result = ClusterInformation.from_dict(data, mock_logger)
        assert result is not None
        assert result.total_cores == 0


class TestClusterInformationFromDlioSummaryJson:
    """Tests for ClusterInformation.from_dlio_summary_json classmethod."""

    def test_from_dlio_summary_json_returns_instance(self, mock_logger):
        """from_dlio_summary_json should return an instance when data is valid."""
        summary = {
            'host_memory_GB': [64, 64],  # Two hosts with 64GB each
            'host_cpu_count': [16, 16],
            'num_hosts': 2
        }
        result = ClusterInformation.from_dlio_summary_json(summary, mock_logger)
        assert isinstance(result, ClusterInformation)

    def test_from_dlio_summary_json_calculates_total_memory(self, mock_logger):
        """from_dlio_summary_json should calculate total memory correctly."""
        summary = {
            'host_memory_GB': [64, 64],  # Two hosts with 64GB each
            'host_cpu_count': [16, 16],
            'num_hosts': 2
        }
        result = ClusterInformation.from_dlio_summary_json(summary, mock_logger)
        expected_bytes = 128 * 1024 * 1024 * 1024  # 128GB in bytes
        assert result.total_memory_bytes == expected_bytes

    def test_from_dlio_summary_json_calculates_total_cores(self, mock_logger):
        """from_dlio_summary_json should calculate total cores correctly."""
        summary = {
            'host_memory_GB': [64, 64],
            'host_cpu_count': [16, 16],
            'num_hosts': 2
        }
        result = ClusterInformation.from_dlio_summary_json(summary, mock_logger)
        assert result.total_cores == 32

    def test_from_dlio_summary_json_returns_none_for_missing_host_memory(self, mock_logger):
        """from_dlio_summary_json should return None if host_memory_GB is missing."""
        summary = {
            'host_cpu_count': [16, 16],
            'num_hosts': 2
        }
        result = ClusterInformation.from_dlio_summary_json(summary, mock_logger)
        assert result is None

    def test_from_dlio_summary_json_returns_none_for_missing_host_cpu_count(self, mock_logger):
        """from_dlio_summary_json should return None if host_cpu_count is missing."""
        summary = {
            'host_memory_GB': [64, 64],
            'num_hosts': 2
        }
        result = ClusterInformation.from_dlio_summary_json(summary, mock_logger)
        assert result is None

    def test_from_dlio_summary_json_returns_none_for_empty_summary(self, mock_logger):
        """from_dlio_summary_json should return None for empty summary."""
        result = ClusterInformation.from_dlio_summary_json({}, mock_logger)
        assert result is None


class TestBenchmarkRunSystemInfoFallback:
    """Tests for BenchmarkRun system_info metadata fallback logic."""

    def test_system_info_from_metadata_when_dlio_summary_lacks_data(self, mock_logger):
        """system_info should fall back to metadata when DLIO summary lacks required data."""
        # Create a mock BenchmarkResult with metadata containing cluster_information
        # but DLIO summary lacking host_memory_GB/host_cpu_count
        mock_benchmark_result = MagicMock(spec=BenchmarkResult)
        mock_benchmark_result.benchmark_result_root_dir = '/tmp/test_run'
        mock_benchmark_result.summary = {
            'workload': {},
            'num_accelerators': 4,
            'start': '2024-01-01T00:00:00',
            'metric': {'throughput': 100}
            # Missing host_memory_GB and host_cpu_count
        }
        mock_benchmark_result.metadata = {
            'cluster_information': {
                'total_memory_bytes': 256 * 1024 * 1024 * 1024,  # 256GB
                'total_cores': 64
            }
        }
        mock_benchmark_result.hydra_configs = {
            'config.yaml': {
                'workload': {
                    'model': {'name': 'unet3d'},
                    'workflow': {'train': True}
                }
            },
            'overrides.yaml': ['workload=training_gpu']
        }

        benchmark_run = BenchmarkRun(benchmark_result=mock_benchmark_result, logger=mock_logger)

        assert benchmark_run.system_info is not None
        assert benchmark_run.system_info.total_memory_bytes == 256 * 1024 * 1024 * 1024
        assert benchmark_run.system_info.total_cores == 64

    def test_system_info_prefers_dlio_summary_when_available(self, mock_logger):
        """system_info should use DLIO summary when it has the required data."""
        mock_benchmark_result = MagicMock(spec=BenchmarkResult)
        mock_benchmark_result.benchmark_result_root_dir = '/tmp/test_run'
        mock_benchmark_result.summary = {
            'workload': {},
            'num_accelerators': 4,
            'start': '2024-01-01T00:00:00',
            'metric': {'throughput': 100},
            'host_memory_GB': [64, 64],  # 128GB total
            'host_cpu_count': [16, 16],  # 32 cores total
            'num_hosts': 2
        }
        mock_benchmark_result.metadata = {
            'cluster_information': {
                'total_memory_bytes': 256 * 1024 * 1024 * 1024,  # Different value
                'total_cores': 64
            }
        }
        mock_benchmark_result.hydra_configs = {
            'config.yaml': {
                'workload': {
                    'model': {'name': 'unet3d'},
                    'workflow': {'train': True}
                }
            },
            'overrides.yaml': ['workload=training_gpu']
        }

        benchmark_run = BenchmarkRun(benchmark_result=mock_benchmark_result, logger=mock_logger)

        # Should use DLIO summary data (128GB), not metadata (256GB)
        expected_bytes = 128 * 1024 * 1024 * 1024
        assert benchmark_run.system_info.total_memory_bytes == expected_bytes

    def test_system_info_none_when_no_data_available(self, mock_logger):
        """system_info should be None when neither DLIO summary nor metadata has data."""
        mock_benchmark_result = MagicMock(spec=BenchmarkResult)
        mock_benchmark_result.benchmark_result_root_dir = '/tmp/test_run'
        mock_benchmark_result.summary = {
            'workload': {},
            'num_accelerators': 4,
            'start': '2024-01-01T00:00:00',
            'metric': {'throughput': 100}
            # Missing host_memory_GB and host_cpu_count
        }
        mock_benchmark_result.metadata = {}  # No cluster_information
        mock_benchmark_result.hydra_configs = {
            'config.yaml': {
                'workload': {
                    'model': {'name': 'unet3d'},
                    'workflow': {'train': True}
                }
            },
            'overrides.yaml': ['workload=training_gpu']
        }

        benchmark_run = BenchmarkRun(benchmark_result=mock_benchmark_result, logger=mock_logger)

        assert benchmark_run.system_info is None


# =============================================================================
# Phase 2 Tests: Extended HostInfo and ClusterInformation
# =============================================================================

class TestHostInfoFromCollectedData:
    """Tests for HostInfo.from_collected_data classmethod."""

    def test_from_collected_data_basic(self, mock_logger):
        """Should create HostInfo from collected data dictionary."""
        from mlpstorage.rules import HostInfo

        data = {
            'hostname': 'node1',
            'collection_timestamp': '2024-01-01T00:00:00Z',
            'meminfo': {
                'MemTotal': 16384000,  # kB
                'MemFree': 8192000,
                'MemAvailable': 10240000,
                'Buffers': 512000,
                'Cached': 1024000,
            },
            'cpuinfo': [
                {'processor': 0, 'model name': 'Intel Xeon', 'physical id': 0, 'core id': 0},
                {'processor': 1, 'model name': 'Intel Xeon', 'physical id': 0, 'core id': 1},
            ],
            'diskstats': [
                {'device_name': 'sda', 'reads_completed': 100, 'writes_completed': 50},
            ],
            'netdev': [
                {'interface_name': 'eth0', 'rx_bytes': 1000, 'tx_bytes': 500},
            ],
            'version': 'Linux version 5.4.0-generic',
            'loadavg': {
                'load_1min': 0.5,
                'load_5min': 0.75,
                'load_15min': 0.8,
                'running_processes': 2,
                'total_processes': 500,
            },
            'uptime_seconds': 86400.0,
            'os_release': {'NAME': 'Ubuntu', 'VERSION_ID': '20.04'},
        }

        host_info = HostInfo.from_collected_data(data)

        assert host_info.hostname == 'node1'
        assert host_info.collection_timestamp == '2024-01-01T00:00:00Z'
        assert host_info.memory.total == 16384000 * 1024  # Converted to bytes
        assert host_info.cpu is not None
        assert host_info.cpu.num_logical_cores == 2
        assert host_info.disks is not None
        assert len(host_info.disks) == 1
        assert host_info.network is not None
        assert len(host_info.network) == 1
        assert host_info.system is not None
        assert host_info.system.uptime_seconds == 86400.0

    def test_from_collected_data_missing_optional_fields(self, mock_logger):
        """Should handle missing optional fields gracefully."""
        from mlpstorage.rules import HostInfo

        data = {
            'hostname': 'node1',
            'meminfo': {'MemTotal': 16384000},
        }

        host_info = HostInfo.from_collected_data(data)

        assert host_info.hostname == 'node1'
        assert host_info.memory.total == 16384000 * 1024
        assert host_info.cpu is None
        assert host_info.disks is None
        assert host_info.network is None

    def test_to_dict_includes_all_fields(self, mock_logger):
        """to_dict should include all populated fields."""
        from mlpstorage.rules import HostInfo

        data = {
            'hostname': 'node1',
            'meminfo': {'MemTotal': 16384000},
            'cpuinfo': [{'processor': 0, 'model name': 'Intel Xeon'}],
        }

        host_info = HostInfo.from_collected_data(data)
        result = host_info.to_dict()

        assert 'hostname' in result
        assert 'memory' in result
        assert 'cpu' in result
        assert result['cpu']['model'] == 'Intel Xeon'


class TestClusterInformationFromMpiCollection:
    """Tests for ClusterInformation.from_mpi_collection classmethod."""

    def test_from_mpi_collection_single_host(self, mock_logger):
        """Should create ClusterInformation from single host collection."""
        collected_data = {
            'node1': {
                'hostname': 'node1',
                'meminfo': {'MemTotal': 16384000},
                'cpuinfo': [
                    {'processor': 0, 'physical id': 0, 'core id': 0},
                    {'processor': 1, 'physical id': 0, 'core id': 1},
                ],
            },
            '_metadata': {
                'collection_method': 'mpi',
                'collection_timestamp': '2024-01-01T00:00:00Z',
            }
        }

        cluster_info = ClusterInformation.from_mpi_collection(collected_data, mock_logger)

        assert cluster_info.num_hosts == 1
        assert cluster_info.total_memory_bytes == 16384000 * 1024
        assert cluster_info.collection_method == 'mpi'
        assert cluster_info.collection_timestamp == '2024-01-01T00:00:00Z'

    def test_from_mpi_collection_multiple_hosts(self, mock_logger):
        """Should aggregate data from multiple hosts."""
        collected_data = {
            'node1': {
                'hostname': 'node1',
                'meminfo': {'MemTotal': 16384000},
                'cpuinfo': [{'processor': 0, 'physical id': 0, 'core id': 0}],
            },
            'node2': {
                'hostname': 'node2',
                'meminfo': {'MemTotal': 32768000},
                'cpuinfo': [
                    {'processor': 0, 'physical id': 0, 'core id': 0},
                    {'processor': 1, 'physical id': 0, 'core id': 1},
                ],
            },
            '_metadata': {
                'collection_method': 'mpi',
            }
        }

        cluster_info = ClusterInformation.from_mpi_collection(collected_data, mock_logger)

        assert cluster_info.num_hosts == 2
        # Total = 16384000 + 32768000 kB = 49152000 kB, converted to bytes
        expected_total = (16384000 + 32768000) * 1024
        assert cluster_info.total_memory_bytes == expected_total
        assert cluster_info.min_memory_bytes == 16384000 * 1024
        assert cluster_info.max_memory_bytes == 32768000 * 1024

    def test_from_mpi_collection_local_fallback(self, mock_logger):
        """Should handle local_fallback collection method."""
        collected_data = {
            'localhost': {
                'hostname': 'localhost',
                'meminfo': {'MemTotal': 16384000},
            },
            '_metadata': {
                'collection_method': 'local_fallback',
                'mpi_error': 'MPI not available',
            }
        }

        cluster_info = ClusterInformation.from_mpi_collection(collected_data, mock_logger)

        assert cluster_info.collection_method == 'local_fallback'


class TestClusterInformationConsistencyValidation:
    """Tests for ClusterInformation.validate_cluster_consistency method."""

    def test_no_issues_for_consistent_cluster(self, mock_logger):
        """Should return empty list for consistent cluster."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo, HostCPUInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=8),
            ),
            HostInfo(
                hostname='node2',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=8),
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        issues = cluster_info.validate_cluster_consistency()

        assert len(issues) == 0

    def test_detects_memory_variance(self, mock_logger):
        """Should detect significant memory variance between hosts."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo, HostCPUInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=8),
            ),
            HostInfo(
                hostname='node2',
                memory=HostMemoryInfo.from_total_mem_int(32 * 1024**3),  # 100% more
                cpu=HostCPUInfo(num_cores=8),
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        issues = cluster_info.validate_cluster_consistency()

        assert len(issues) == 1
        assert 'Memory variance' in issues[0]

    def test_detects_cpu_core_variance(self, mock_logger):
        """Should detect CPU core count variance."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo, HostCPUInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=8),
            ),
            HostInfo(
                hostname='node2',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=16),  # Different core count
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        issues = cluster_info.validate_cluster_consistency()

        assert any('CPU core count' in issue for issue in issues)

    def test_single_host_no_validation(self, mock_logger):
        """Should skip validation for single-host clusters."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        issues = cluster_info.validate_cluster_consistency()

        assert len(issues) == 0


class TestClusterInformationExtendedAsDict:
    """Tests for extended ClusterInformation.as_dict method."""

    def test_as_dict_includes_new_fields(self, mock_logger):
        """as_dict should include all new fields."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        cluster_info.collection_method = 'mpi'
        cluster_info.collection_timestamp = '2024-01-01T00:00:00Z'

        result = cluster_info.as_dict()

        assert 'num_hosts' in result
        assert result['num_hosts'] == 1
        assert 'min_memory_bytes' in result
        assert 'max_memory_bytes' in result
        assert 'collection_method' in result
        assert result['collection_method'] == 'mpi'
        assert 'hosts' in result

    def test_as_dict_includes_consistency_issues(self, mock_logger):
        """as_dict should include consistency issues if present."""
        from mlpstorage.rules import HostInfo, HostMemoryInfo, HostCPUInfo

        host_info_list = [
            HostInfo(
                hostname='node1',
                memory=HostMemoryInfo.from_total_mem_int(16 * 1024**3),
                cpu=HostCPUInfo(num_cores=8),
            ),
            HostInfo(
                hostname='node2',
                memory=HostMemoryInfo.from_total_mem_int(32 * 1024**3),
                cpu=HostCPUInfo(num_cores=16),
            ),
        ]

        cluster_info = ClusterInformation(host_info_list, mock_logger)
        cluster_info.validate_cluster_consistency()

        result = cluster_info.as_dict()

        assert 'host_consistency_issues' in result
        assert len(result['host_consistency_issues']) > 0


class TestClusterInformationFromDictExtended:
    """Tests for extended from_dict loading."""

    def test_from_dict_restores_extended_fields(self, mock_logger):
        """from_dict should restore all extended fields."""
        data = {
            'total_memory_bytes': 32 * 1024**3,
            'total_cores': 16,
            'num_hosts': 2,
            'min_memory_bytes': 16 * 1024**3,
            'max_memory_bytes': 16 * 1024**3,
            'collection_method': 'mpi',
            'collection_timestamp': '2024-01-01T00:00:00Z',
            'host_consistency_issues': ['Test issue'],
        }

        cluster_info = ClusterInformation.from_dict(data, mock_logger)

        assert cluster_info.num_hosts == 2
        assert cluster_info.min_memory_bytes == 16 * 1024**3
        assert cluster_info.max_memory_bytes == 16 * 1024**3
        assert cluster_info.collection_method == 'mpi'
        assert cluster_info.collection_timestamp == '2024-01-01T00:00:00Z'
        assert len(cluster_info.host_consistency_issues) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
