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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
