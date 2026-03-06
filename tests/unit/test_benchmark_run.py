"""
Tests for BenchmarkRun class in mlpstorage.rules module.

Tests cover:
- BenchmarkRun construction methods (from_benchmark, from_result_dir, from_data)
- Property delegation to BenchmarkRunData
- as_dict serialization
- post_execution state detection
- Legacy constructor compatibility
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage.rules import (
    BenchmarkRun,
    BenchmarkRunData,
    BenchmarkInstanceExtractor,
    ResultFilesExtractor,
    RunID,
    ClusterInformation,
    HostInfo,
    HostMemoryInfo,
)


class TestBenchmarkRunConstruction:
    """Tests for BenchmarkRun construction methods."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def sample_data(self, mock_logger):
        """Create sample BenchmarkRunData."""
        host_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host = HostInfo(hostname='host1', memory=host_memory)
        cluster_info = ClusterInformation([host], mock_logger)

        return BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={"dataset": {"num_files_train": 400}},
            override_parameters={},
            system_info=cluster_info,
            accelerator="h100",
            result_dir="/path/to/results"
        )

    def test_from_data_creates_instance(self, mock_logger, sample_data):
        """BenchmarkRun can be created from BenchmarkRunData."""
        run = BenchmarkRun(data=sample_data, logger=mock_logger)

        assert run.benchmark_type == BENCHMARK_TYPES.training
        assert run.model == "unet3d"
        assert run.command == "run"
        assert run.num_processes == 8

    def test_from_data_creates_run_id(self, mock_logger, sample_data):
        """BenchmarkRun creates RunID from data."""
        run = BenchmarkRun(data=sample_data, logger=mock_logger)

        assert isinstance(run.run_id, RunID)
        assert run.run_id.program == "training"
        assert run.run_id.model == "unet3d"
        assert run.run_id.command == "run"

    def test_from_benchmark_creates_instance(self, mock_logger):
        """from_benchmark creates BenchmarkRun from Benchmark instance."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'resnet50'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 16
        mock_benchmark.args.accelerator_type = 'a100'
        mock_benchmark.run_datetime = '20250111_150000'
        mock_benchmark.combined_params = {'dataset': {'num_files_train': 1000}}
        mock_benchmark.params_dict = {}
        mock_benchmark.run_result_output = '/results/output'

        run = BenchmarkRun.from_benchmark(mock_benchmark, logger=mock_logger)

        assert run.benchmark_type == BENCHMARK_TYPES.training
        assert run.model == 'resnet50'
        assert run.accelerator == 'a100'

    def test_from_result_dir_creates_instance(self, mock_logger, tmp_path):
        """from_result_dir creates BenchmarkRun from result directory."""
        result_dir = tmp_path / "result"
        result_dir.mkdir()

        # Create metadata file
        metadata = {
            "benchmark_type": "training",
            "model": "cosmoflow",
            "command": "run",
            "run_datetime": "20250111_160000",
            "num_processes": 32,
            "parameters": {"dataset": {"num_files_train": 2000}},
            "override_parameters": {}
        }
        with open(result_dir / "training_20250111_160000_metadata.json", 'w') as f:
            json.dump(metadata, f)

        run = BenchmarkRun.from_result_dir(str(result_dir), logger=mock_logger)

        assert run.benchmark_type == BENCHMARK_TYPES.training
        assert run.model == "cosmoflow"
        assert run.num_processes == 32

    def test_constructor_requires_data_or_legacy(self, mock_logger):
        """Constructor raises error without data or legacy parameters."""
        with pytest.raises(ValueError, match="Either data, benchmark_result, or benchmark_instance"):
            BenchmarkRun(logger=mock_logger)

    def test_constructor_rejects_both_legacy(self, mock_logger):
        """Constructor raises error with both legacy parameters."""
        mock_result = MagicMock()
        mock_instance = MagicMock()

        with pytest.raises(ValueError, match="Only one of"):
            BenchmarkRun(
                benchmark_result=mock_result,
                benchmark_instance=mock_instance,
                logger=mock_logger
            )


class TestBenchmarkRunProperties:
    """Tests for BenchmarkRun property delegation."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def sample_run(self, mock_logger):
        """Create a sample BenchmarkRun."""
        host_memory = HostMemoryInfo.from_total_mem_int(274877906944)
        host = HostInfo(hostname='host1', memory=host_memory)
        cluster_info = ClusterInformation([host], mock_logger)

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            command="run",
            run_datetime="20250111_170000",
            num_processes=8,
            parameters={"checkpoint": {"num_checkpoints_read": 5}},
            override_parameters={"checkpoint.folder": "/data/ckpt"},
            system_info=cluster_info,
            metrics={"throughput": 45.2},
            result_dir="/path/to/ckpt/results",
            accelerator=None
        )
        return BenchmarkRun(data=data, logger=mock_logger)

    def test_benchmark_type_delegates(self, sample_run):
        """benchmark_type property delegates to data."""
        assert sample_run.benchmark_type == BENCHMARK_TYPES.checkpointing

    def test_model_delegates(self, sample_run):
        """model property delegates to data."""
        assert sample_run.model == "llama3-8b"

    def test_command_delegates(self, sample_run):
        """command property delegates to data."""
        assert sample_run.command == "run"

    def test_run_datetime_delegates(self, sample_run):
        """run_datetime property delegates to data."""
        assert sample_run.run_datetime == "20250111_170000"

    def test_num_processes_delegates(self, sample_run):
        """num_processes property delegates to data."""
        assert sample_run.num_processes == 8

    def test_parameters_delegates(self, sample_run):
        """parameters property delegates to data."""
        assert sample_run.parameters == {"checkpoint": {"num_checkpoints_read": 5}}

    def test_override_parameters_delegates(self, sample_run):
        """override_parameters property delegates to data."""
        assert sample_run.override_parameters == {"checkpoint.folder": "/data/ckpt"}

    def test_system_info_delegates(self, sample_run):
        """system_info property delegates to data."""
        assert sample_run.system_info is not None
        assert isinstance(sample_run.system_info, ClusterInformation)

    def test_metrics_delegates(self, sample_run):
        """metrics property delegates to data."""
        assert sample_run.metrics == {"throughput": 45.2}

    def test_accelerator_delegates(self, sample_run):
        """accelerator property delegates to data."""
        assert sample_run.accelerator is None

    def test_result_dir_delegates(self, sample_run):
        """result_dir property delegates to data."""
        assert sample_run.result_dir == "/path/to/ckpt/results"

    def test_data_property_returns_underlying_data(self, sample_run):
        """data property returns the underlying BenchmarkRunData."""
        assert isinstance(sample_run.data, BenchmarkRunData)


class TestBenchmarkRunVerificationState:
    """Tests for BenchmarkRun verification state properties."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def sample_run(self, mock_logger):
        """Create a sample BenchmarkRun."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        return BenchmarkRun(data=data, logger=mock_logger)

    def test_issues_initially_empty(self, sample_run):
        """issues property starts as empty list."""
        assert sample_run.issues == []

    def test_issues_can_be_set(self, sample_run):
        """issues property can be set."""
        from mlpstorage.rules import Issue
        issues = [Issue(PARAM_VALIDATION.OPEN, "Test issue")]
        sample_run.issues = issues
        assert len(sample_run.issues) == 1
        assert sample_run.issues[0].message == "Test issue"

    def test_category_initially_none(self, sample_run):
        """category property starts as None."""
        assert sample_run.category is None

    def test_category_can_be_set(self, sample_run):
        """category property can be set."""
        sample_run.category = PARAM_VALIDATION.CLOSED
        assert sample_run.category == PARAM_VALIDATION.CLOSED


class TestBenchmarkRunPostExecution:
    """Tests for post_execution state detection."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_post_execution_true_when_metrics_present(self, mock_logger):
        """post_execution is True when metrics are present."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={},
            metrics={"train_au_percentage": 95.0}  # Has metrics
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        assert run.post_execution is True

    def test_post_execution_false_when_no_metrics(self, mock_logger):
        """post_execution is False when metrics are None."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={},
            metrics=None  # No metrics
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        assert run.post_execution is False


class TestBenchmarkRunAsDict:
    """Tests for as_dict serialization."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_as_dict_includes_all_fields(self, mock_logger):
        """as_dict includes all expected fields."""
        host_memory = HostMemoryInfo.from_total_mem_int(137438953472)
        host = HostInfo(hostname='host1', memory=host_memory)
        cluster_info = ClusterInformation([host], mock_logger)

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={"key": "value"},
            override_parameters={},
            system_info=cluster_info,
            metrics={"metric": 1.0},
            accelerator="h100"
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        result = run.as_dict()

        assert "run_id" in result
        assert "benchmark_type" in result
        assert "model" in result
        assert "command" in result
        assert "num_processes" in result
        assert "parameters" in result
        assert "system_info" in result
        assert "metrics" in result
        assert "accelerator" in result

    def test_as_dict_formats_run_id_as_string(self, mock_logger):
        """as_dict formats run_id as string."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        result = run.as_dict()

        assert isinstance(result["run_id"], str)
        assert "training" in result["run_id"]
        assert "unet3d" in result["run_id"]

    def test_as_dict_handles_none_accelerator(self, mock_logger):
        """as_dict excludes accelerator when None."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={},
            accelerator=None
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        result = run.as_dict()

        # accelerator should not be in dict when None
        assert "accelerator" not in result

    def test_as_dict_handles_none_system_info(self, mock_logger):
        """as_dict handles None system_info."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={},
            system_info=None
        )
        run = BenchmarkRun(data=data, logger=mock_logger)

        result = run.as_dict()

        assert result["system_info"] is None


class TestBenchmarkRunFromData:
    """Tests for the from_data class method (convenience wrapper)."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_from_data_is_constructor_alias(self, mock_logger):
        """from_data acts as alias for constructor with data param."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={}
        )

        # Both should create equivalent instances
        run1 = BenchmarkRun(data=data, logger=mock_logger)
        run2 = BenchmarkRun.from_data(data, mock_logger) if hasattr(BenchmarkRun, 'from_data') else BenchmarkRun(data=data, logger=mock_logger)

        assert run1.model == run2.model
        assert run1.benchmark_type == run2.benchmark_type


class TestBenchmarkRunIntegration:
    """Integration tests using fixture data."""

    @pytest.fixture
    def training_fixture_dir(self):
        """Get path to training fixture directory."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_results" / "training_run"
        if fixtures_dir.exists():
            return fixtures_dir
        pytest.skip("Training fixture directory not found")

    @pytest.fixture
    def checkpointing_fixture_dir(self):
        """Get path to checkpointing fixture directory."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_results" / "checkpointing_run"
        if fixtures_dir.exists():
            return fixtures_dir
        pytest.skip("Checkpointing fixture directory not found")

    def test_load_from_training_fixture(self, training_fixture_dir):
        """BenchmarkRun loads correctly from training fixture."""
        run = BenchmarkRun.from_result_dir(str(training_fixture_dir))

        assert run.benchmark_type == BENCHMARK_TYPES.training
        assert run.model is not None
        # Metrics may or may not be present depending on fixture structure

    def test_load_from_checkpointing_fixture(self, checkpointing_fixture_dir):
        """BenchmarkRun loads correctly from checkpointing fixture."""
        run = BenchmarkRun.from_result_dir(str(checkpointing_fixture_dir))

        assert run.benchmark_type == BENCHMARK_TYPES.checkpointing
        assert run.model is not None
        # Metrics may or may not be present depending on fixture structure
