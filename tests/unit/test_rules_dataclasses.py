"""
Tests for dataclasses in mlpstorage.rules module.

Tests cover:
- Issue dataclass and string representation
- RunID dataclass and string format
- ProcessedRun validity checking
- HostMemoryInfo factory methods
- HostCPUInfo factory methods
- HostInfo factory methods
- ClusterInformation aggregation
- BenchmarkRunData structure
"""

import pytest
from unittest.mock import MagicMock

from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES
from mlpstorage.rules import (
    Issue,
    RunID,
    ProcessedRun,
    BenchmarkRunData,
    HostMemoryInfo,
    HostCPUInfo,
    HostInfo,
    ClusterInformation,
)


class TestIssue:
    """Tests for Issue dataclass."""

    def test_str_with_all_fields(self):
        """Issue __str__ includes all fields when present."""
        issue = Issue(
            validation=PARAM_VALIDATION.INVALID,
            message="Parameter out of range",
            parameter="batch_size",
            expected=32,
            actual=64
        )
        result = str(issue)
        assert "INVALID" in result
        assert "Parameter out of range" in result
        assert "batch_size" in result
        assert "32" in result
        assert "64" in result

    def test_str_without_expected_actual(self):
        """Issue __str__ works without expected/actual."""
        issue = Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="Non-standard parameter used",
            parameter="custom_flag"
        )
        result = str(issue)
        assert "OPEN" in result
        assert "Non-standard parameter used" in result
        assert "custom_flag" in result
        assert "Expected" not in result

    def test_str_without_parameter(self):
        """Issue __str__ works without parameter."""
        issue = Issue(
            validation=PARAM_VALIDATION.CLOSED,
            message="All parameters valid"
        )
        result = str(issue)
        assert "CLOSED" in result
        assert "All parameters valid" in result
        assert "Parameter:" not in result

    def test_default_severity(self):
        """Issue has default severity of 'error'."""
        issue = Issue(
            validation=PARAM_VALIDATION.INVALID,
            message="Test"
        )
        assert issue.severity == "error"

    def test_custom_severity(self):
        """Issue accepts custom severity."""
        issue = Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="Test",
            severity="warning"
        )
        assert issue.severity == "warning"


class TestRunID:
    """Tests for RunID dataclass."""

    def test_str_format_complete(self):
        """RunID __str__ formats all components correctly."""
        run_id = RunID(
            program="training",
            command="run",
            model="unet3d",
            run_datetime="20250111_143022"
        )
        result = str(run_id)
        assert result == "training_run_unet3d_20250111_143022"

    def test_str_without_command(self):
        """RunID __str__ handles empty command."""
        run_id = RunID(
            program="checkpointing",
            command="",
            model="llama3-8b",
            run_datetime="20250111_143022"
        )
        result = str(run_id)
        # With empty command, just program_model_datetime
        assert "checkpointing" in result
        assert "llama3-8b" in result
        assert "20250111_143022" in result

    def test_str_without_model(self):
        """RunID __str__ handles empty model."""
        run_id = RunID(
            program="reports",
            command="generate",
            model="",
            run_datetime="20250111_143022"
        )
        result = str(run_id)
        assert "reports" in result
        assert "generate" in result

    def test_str_with_none_values(self):
        """RunID __str__ handles None values gracefully."""
        run_id = RunID(
            program="training",
            command=None,
            model=None,
            run_datetime="20250111_143022"
        )
        # None values should not appear in string
        result = str(run_id)
        assert "training" in result
        assert "20250111_143022" in result


class TestProcessedRun:
    """Tests for ProcessedRun dataclass."""

    def test_is_valid_with_no_issues(self):
        """ProcessedRun is valid when there are no issues."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[]
        )
        assert run.is_valid() is True

    def test_is_valid_with_open_issues(self):
        """ProcessedRun is valid with only OPEN issues."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.OPEN, "Non-standard param used")
            ]
        )
        assert run.is_valid() is True

    def test_is_valid_with_closed_issues(self):
        """ProcessedRun is valid with only CLOSED issues."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.CLOSED, "Parameter validated")
            ]
        )
        assert run.is_valid() is True

    def test_is_valid_with_invalid_issue(self):
        """ProcessedRun is not valid with INVALID issue."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.INVALID, "Invalid parameter value")
            ]
        )
        assert run.is_valid() is False

    def test_is_closed_with_no_issues(self):
        """ProcessedRun is closed when there are no issues."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[]
        )
        assert run.is_closed() is True

    def test_is_closed_with_closed_issues(self):
        """ProcessedRun is closed with only CLOSED issues."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.CLOSED, "Parameter validated")
            ]
        )
        assert run.is_closed() is True

    def test_is_closed_with_open_issue(self):
        """ProcessedRun is not closed with OPEN issue."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.OPEN, "Non-standard param")
            ]
        )
        assert run.is_closed() is False

    def test_is_closed_with_invalid_issue(self):
        """ProcessedRun is not closed with INVALID issue."""
        run = ProcessedRun(
            run_id=RunID("training", "run", "unet3d", "20250111_143022"),
            benchmark_type="training",
            run_parameters={},
            run_metrics={},
            issues=[
                Issue(PARAM_VALIDATION.INVALID, "Invalid param")
            ]
        )
        assert run.is_closed() is False


class TestBenchmarkRunData:
    """Tests for BenchmarkRunData dataclass."""

    def test_creation_with_required_fields(self):
        """BenchmarkRunData can be created with required fields."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={"dataset": {"num_files_train": 400}},
            override_parameters={}
        )
        assert data.benchmark_type == BENCHMARK_TYPES.training
        assert data.model == "unet3d"
        assert data.num_processes == 8

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        assert data.system_info is None
        assert data.metrics is None
        assert data.result_dir is None
        assert data.accelerator is None

    def test_all_fields_can_be_set(self):
        """All fields can be explicitly set."""
        mock_cluster_info = MagicMock()
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="resnet50",
            command="run",
            run_datetime="20250111_143022",
            num_processes=16,
            parameters={"key": "value"},
            override_parameters={"override": "value"},
            system_info=mock_cluster_info,
            metrics={"throughput": 1000.0},
            result_dir="/path/to/results",
            accelerator="h100"
        )
        assert data.system_info is mock_cluster_info
        assert data.metrics == {"throughput": 1000.0}
        assert data.result_dir == "/path/to/results"
        assert data.accelerator == "h100"


class TestHostMemoryInfo:
    """Tests for HostMemoryInfo dataclass."""

    def test_from_psutil_dict(self):
        """HostMemoryInfo.from_psutil_dict creates correct instance."""
        psutil_data = {
            'total': 137438953472,  # 128 GB
            'available': 68719476736,  # 64 GB
            'used': 68719476736,
            'free': 17179869184,
            'active': 51539607552,
            'inactive': 17179869184,
            'buffers': 4294967296,
            'cached': 12884901888,
            'shared': 1073741824
        }
        result = HostMemoryInfo.from_psutil_dict(psutil_data)
        assert result.total == 137438953472
        assert result.available == 68719476736
        assert result.used == 68719476736
        assert result.buffers == 4294967296

    def test_from_psutil_dict_missing_fields(self):
        """HostMemoryInfo.from_psutil_dict handles missing fields."""
        psutil_data = {
            'total': 137438953472
        }
        result = HostMemoryInfo.from_psutil_dict(psutil_data)
        assert result.total == 137438953472
        assert result.available == 0
        assert result.used == 0

    def test_from_total_mem_int(self):
        """HostMemoryInfo.from_total_mem_int creates instance with total only."""
        total_bytes = 137438953472  # 128 GB
        result = HostMemoryInfo.from_total_mem_int(total_bytes)
        assert result.total == total_bytes
        assert result.available is None
        assert result.used is None
        assert result.free is None

    def test_direct_construction(self):
        """HostMemoryInfo can be constructed directly."""
        info = HostMemoryInfo(
            total=137438953472,
            available=68719476736,
            used=68719476736,
            free=17179869184,
            active=51539607552,
            inactive=17179869184,
            buffers=4294967296,
            cached=12884901888,
            shared=1073741824
        )
        assert info.total == 137438953472


class TestHostCPUInfo:
    """Tests for HostCPUInfo dataclass."""

    def test_from_dict(self):
        """HostCPUInfo.from_dict creates correct instance."""
        cpu_data = {
            'num_cores': 32,
            'num_logical_cores': 64,
            'model': 'Intel Xeon',
            'architecture': 'x86_64'
        }
        result = HostCPUInfo.from_dict(cpu_data)
        assert result.num_cores == 32
        assert result.num_logical_cores == 64
        assert result.model == 'Intel Xeon'
        assert result.architecture == 'x86_64'

    def test_from_dict_missing_fields(self):
        """HostCPUInfo.from_dict handles missing fields."""
        cpu_data = {'num_cores': 16}
        result = HostCPUInfo.from_dict(cpu_data)
        assert result.num_cores == 16
        assert result.num_logical_cores == 0
        assert result.model == ""
        assert result.architecture == ""

    def test_from_dict_empty(self):
        """HostCPUInfo.from_dict handles empty dict."""
        result = HostCPUInfo.from_dict({})
        assert result.num_cores == 0
        assert result.num_logical_cores == 0

    def test_default_values(self):
        """HostCPUInfo has correct default values."""
        info = HostCPUInfo()
        assert info.num_cores == 0
        assert info.num_logical_cores == 0
        assert info.model == ""
        assert info.architecture == ""


class TestHostInfo:
    """Tests for HostInfo dataclass."""

    def test_from_dict_with_psutil_memory(self):
        """HostInfo.from_dict detects and uses psutil format."""
        data = {
            'memory_info': {
                'total': 137438953472,
                'available': 68719476736,
                'used': 68719476736,
                'free': 17179869184
            },
            'cpu_info': {
                'num_cores': 32,
                'num_logical_cores': 64
            }
        }
        result = HostInfo.from_dict('host1', data)
        assert result.hostname == 'host1'
        assert result.memory.total == 137438953472
        assert result.cpu.num_cores == 32

    def test_from_dict_with_empty_memory(self):
        """HostInfo.from_dict handles empty memory info."""
        data = {
            'memory_info': {},
            'cpu_info': {'num_cores': 16}
        }
        result = HostInfo.from_dict('host2', data)
        assert result.hostname == 'host2'
        # Should get default memory info
        assert result.cpu.num_cores == 16

    def test_from_dict_without_cpu_info(self):
        """HostInfo.from_dict handles missing cpu_info."""
        data = {
            'memory_info': {
                'total': 137438953472
            }
        }
        result = HostInfo.from_dict('host3', data)
        assert result.hostname == 'host3'
        assert result.cpu is None


class TestClusterInformation:
    """Tests for ClusterInformation class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_aggregates_memory_from_hosts(self, mock_logger):
        """ClusterInformation aggregates memory from all hosts."""
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)  # 128 GB
        host2_memory = HostMemoryInfo.from_total_mem_int(137438953472)  # 128 GB

        host1 = HostInfo(hostname='host1', memory=host1_memory)
        host2 = HostInfo(hostname='host2', memory=host2_memory)

        cluster = ClusterInformation([host1, host2], mock_logger)
        assert cluster.total_memory_bytes == 274877906944  # 256 GB total

    def test_aggregates_cores_from_hosts(self, mock_logger):
        """ClusterInformation aggregates CPU cores from all hosts."""
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host1_cpu = HostCPUInfo(num_cores=32)
        host2_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host2_cpu = HostCPUInfo(num_cores=64)

        host1 = HostInfo(hostname='host1', memory=host1_memory, cpu=host1_cpu)
        host2 = HostInfo(hostname='host2', memory=host2_memory, cpu=host2_cpu)

        cluster = ClusterInformation([host1, host2], mock_logger)
        assert cluster.total_cores == 96

    def test_skip_aggregation(self, mock_logger):
        """ClusterInformation can skip aggregation calculation."""
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host1 = HostInfo(hostname='host1', memory=host1_memory)

        cluster = ClusterInformation([host1], mock_logger, calculate_aggregated_info=False)
        assert cluster.total_memory_bytes == 0
        assert cluster.total_cores == 0

    def test_as_dict(self, mock_logger):
        """ClusterInformation.as_dict returns correct structure."""
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host1_cpu = HostCPUInfo(num_cores=32)
        host1 = HostInfo(hostname='host1', memory=host1_memory, cpu=host1_cpu)

        cluster = ClusterInformation([host1], mock_logger)
        result = cluster.as_dict()

        assert 'total_memory_bytes' in result
        assert 'total_cores' in result
        assert result['total_memory_bytes'] == 137438953472
        assert result['total_cores'] == 32

    def test_from_dlio_summary_json(self, mock_logger):
        """ClusterInformation.from_dlio_summary_json parses correctly."""
        summary = {
            'host_memory_GB': [256, 256],  # Two hosts with 256 GB each
            'host_cpu_count': [64, 64],
            'num_hosts': 2
        }
        cluster = ClusterInformation.from_dlio_summary_json(summary, mock_logger)

        # Total should be 512 GB in bytes
        expected_memory = 512 * 1024 * 1024 * 1024
        assert cluster.total_memory_bytes == expected_memory
        assert cluster.total_cores == 128

    def test_handles_hosts_without_cpu(self, mock_logger):
        """ClusterInformation handles hosts without CPU info."""
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host1 = HostInfo(hostname='host1', memory=host1_memory, cpu=None)

        cluster = ClusterInformation([host1], mock_logger)
        assert cluster.total_memory_bytes == 137438953472
        assert cluster.total_cores == 0

    def test_empty_host_list(self, mock_logger):
        """ClusterInformation handles empty host list."""
        cluster = ClusterInformation([], mock_logger)
        assert cluster.total_memory_bytes == 0
        assert cluster.total_cores == 0
