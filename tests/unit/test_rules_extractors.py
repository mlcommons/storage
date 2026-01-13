"""
Tests for extractor classes in mlpstorage.rules module.

Tests cover:
- BenchmarkInstanceExtractor: extracting data from live Benchmark instances
- DLIOResultParser: parsing DLIO-specific result files
- ResultFilesExtractor: orchestrating extraction from result directories
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mlpstorage.config import BENCHMARK_TYPES
from mlpstorage.rules import (
    BenchmarkInstanceExtractor,
    DLIOResultParser,
    ResultFilesExtractor,
    BenchmarkRunData,
    ClusterInformation,
)


class TestBenchmarkInstanceExtractor:
    """Tests for BenchmarkInstanceExtractor class."""

    def test_extract_with_all_attributes(self):
        """Extract from benchmark with all expected attributes."""
        # Create mock benchmark with all attributes
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 8
        mock_benchmark.args.accelerator_type = 'h100'
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.combined_params = {'dataset': {'num_files_train': 400}}
        mock_benchmark.params_dict = {'read_threads': 8}
        mock_benchmark.cluster_information = MagicMock()
        mock_benchmark.run_result_output = '/path/to/results'

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert isinstance(result, BenchmarkRunData)
        assert result.benchmark_type == BENCHMARK_TYPES.training
        assert result.model == 'unet3d'
        assert result.command == 'run'
        assert result.num_processes == 8
        assert result.accelerator == 'h100'
        assert result.run_datetime == '20250111_143022'
        assert result.parameters == {'dataset': {'num_files_train': 400}}
        assert result.override_parameters == {'read_threads': 8}
        assert result.result_dir == '/path/to/results'

    def test_extract_without_combined_params(self):
        """Extract handles benchmark without combined_params."""
        mock_benchmark = MagicMock(spec=['BENCHMARK_TYPE', 'args', 'run_datetime', 'params_dict'])
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'resnet50'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 4
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.params_dict = {}
        # No combined_params attribute

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert result.parameters == {}

    def test_extract_without_params_dict(self):
        """Extract handles benchmark without params_dict."""
        mock_benchmark = MagicMock(spec=['BENCHMARK_TYPE', 'args', 'run_datetime', 'combined_params'])
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.checkpointing
        mock_benchmark.args.model = 'llama3-8b'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 8
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.combined_params = {}
        # No params_dict attribute

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert result.override_parameters == {}

    def test_extract_without_cluster_information(self):
        """Extract handles benchmark without cluster_information."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'cosmoflow'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 16
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.combined_params = {}
        mock_benchmark.params_dict = {}
        # Remove cluster_information
        del mock_benchmark.cluster_information

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert result.system_info is None

    def test_extract_without_run_result_output(self):
        """Extract handles benchmark without run_result_output."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 8
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.combined_params = {}
        mock_benchmark.params_dict = {}
        # Remove run_result_output
        del mock_benchmark.run_result_output

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert result.result_dir is None

    def test_extract_metrics_is_none(self):
        """Extract always sets metrics to None (pre-execution)."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'
        mock_benchmark.args.num_processes = 8
        mock_benchmark.run_datetime = '20250111_143022'
        mock_benchmark.combined_params = {}
        mock_benchmark.params_dict = {}

        result = BenchmarkInstanceExtractor.extract(mock_benchmark)

        assert result.metrics is None


class TestDLIOResultParser:
    """Tests for DLIOResultParser class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def training_result_dir(self, tmp_path):
        """Create a mock training result directory."""
        result_dir = tmp_path / "training_run"
        result_dir.mkdir()

        # Create summary.json
        summary = {
            "start": "2025-01-11 14:30:22",
            "end": "2025-01-11 14:35:22",
            "num_accelerators": 8,
            "num_hosts": 2,
            "host_memory_GB": [256, 256],
            "host_cpu_count": [64, 64],
            "workload": "unet3d_h100",
            "metric": {
                "train_au_percentage": [95.2, 94.8, 95.1],
                "train_throughput_samples_per_second": [1250.5, 1248.2, 1251.0]
            }
        }
        with open(result_dir / "summary.json", 'w') as f:
            json.dump(summary, f)

        # Create .hydra directory with config files
        hydra_dir = result_dir / "dlio_config"
        hydra_dir.mkdir()

        config = {
            "workload": {
                "model": {"name": "unet3d"},
                "dataset": {"num_files_train": 400, "data_folder": "/data"},
                "workflow": {
                    "generate_data": False,
                    "train": True,
                    "checkpoint": False
                }
            }
        }
        with open(hydra_dir / "config.yaml", 'w') as f:
            import yaml
            yaml.dump(config, f)

        overrides = [
            "workload=unet3d_h100",
            "++workload.dataset.num_files_train=400"
        ]
        with open(hydra_dir / "overrides.yaml", 'w') as f:
            yaml.dump(overrides, f)

        return result_dir

    @pytest.fixture
    def checkpointing_result_dir(self, tmp_path):
        """Create a mock checkpointing result directory."""
        result_dir = tmp_path / "checkpointing_run"
        result_dir.mkdir()

        # Create summary.json
        summary = {
            "start": "2025-01-11 15:00:00",
            "end": "2025-01-11 15:05:30",
            "num_accelerators": 8,
            "num_hosts": 1,
            "host_memory_GB": [512],
            "host_cpu_count": [64],
            "workload": "llama3_8b",
            "metric": {
                "checkpoint_write_throughput_GB_per_second": [45.2, 44.8, 45.0],
                "checkpoint_read_throughput_GB_per_second": [52.1, 51.8, 52.0]
            }
        }
        with open(result_dir / "summary.json", 'w') as f:
            json.dump(summary, f)

        # Create .hydra directory with config files
        hydra_dir = result_dir / "dlio_config"
        hydra_dir.mkdir()

        config = {
            "workload": {
                "model": {"name": "llama3_8b"},
                "checkpoint": {
                    "checkpoint_folder": "/data/checkpoints",
                    "num_checkpoints_read": 1,
                    "num_checkpoints_write": 1
                },
                "workflow": {
                    "generate_data": False,
                    "train": False,
                    "checkpoint": True
                }
            }
        }
        with open(hydra_dir / "config.yaml", 'w') as f:
            import yaml
            yaml.dump(config, f)

        overrides = [
            "workload=llama3_8b",
            "++workload.checkpoint.checkpoint_folder=/data/checkpoints"
        ]
        with open(hydra_dir / "overrides.yaml", 'w') as f:
            yaml.dump(overrides, f)

        return result_dir

    def test_parse_training_run(self, mock_logger, training_result_dir):
        """DLIOResultParser correctly parses training run."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(training_result_dir))

        assert isinstance(result, BenchmarkRunData)
        assert result.benchmark_type == BENCHMARK_TYPES.training
        assert result.model == "unet3d"
        assert result.command == "run"
        assert result.num_processes == 8
        assert result.metrics is not None
        assert 'train_au_percentage' in result.metrics

    def test_parse_checkpointing_run(self, mock_logger, checkpointing_result_dir):
        """DLIOResultParser correctly parses checkpointing run."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(checkpointing_result_dir))

        assert result.benchmark_type == BENCHMARK_TYPES.checkpointing
        assert result.model == "llama3-8b"  # Note: normalized name
        assert result.command == "run"
        assert result.metrics is not None

    def test_parse_extracts_override_params(self, mock_logger, training_result_dir):
        """DLIOResultParser extracts override parameters from Hydra overrides."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(training_result_dir))

        # Should extract ++workload.* params
        assert 'dataset.num_files_train' in result.override_parameters

    def test_parse_missing_summary_raises_error(self, mock_logger, tmp_path):
        """DLIOResultParser raises error when summary.json is missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        parser = DLIOResultParser(logger=mock_logger)
        with pytest.raises(ValueError, match="No summary.json found"):
            parser.parse(str(empty_dir))

    def test_parse_builds_system_info(self, mock_logger, training_result_dir):
        """DLIOResultParser builds ClusterInformation from summary."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(training_result_dir))

        assert result.system_info is not None
        assert isinstance(result.system_info, ClusterInformation)
        # 256 + 256 = 512 GB in bytes
        expected_memory = 512 * 1024 * 1024 * 1024
        assert result.system_info.total_memory_bytes == expected_memory

    def test_parse_normalizes_llama_model_name(self, mock_logger, checkpointing_result_dir):
        """DLIOResultParser normalizes llama model names."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(checkpointing_result_dir))

        # llama3_8b should become llama3-8b
        assert result.model == "llama3-8b"

    def test_parse_extracts_accelerator(self, mock_logger, training_result_dir):
        """DLIOResultParser extracts accelerator from workload override."""
        parser = DLIOResultParser(logger=mock_logger)
        result = parser.parse(str(training_result_dir))

        # From workload=unet3d_h100, should extract h100
        assert result.accelerator == "h100"


class TestResultFilesExtractor:
    """Tests for ResultFilesExtractor class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def complete_metadata_dir(self, tmp_path):
        """Create a result directory with complete metadata."""
        result_dir = tmp_path / "complete_metadata"
        result_dir.mkdir()

        metadata = {
            "benchmark_type": "training",
            "model": "unet3d",
            "command": "run",
            "run_datetime": "20250111_143022",
            "num_processes": 8,
            "accelerator": "h100",
            "parameters": {
                "dataset": {"num_files_train": 400}
            },
            "override_parameters": {}
        }
        with open(result_dir / "training_20250111_143022_metadata.json", 'w') as f:
            json.dump(metadata, f)

        return result_dir

    @pytest.fixture
    def incomplete_metadata_dir(self, tmp_path):
        """Create a result directory with incomplete metadata."""
        result_dir = tmp_path / "incomplete_metadata"
        result_dir.mkdir()

        # Incomplete metadata - missing required fields
        metadata = {
            "model": "unet3d",
            "command": "run"
        }
        with open(result_dir / "training_20250111_143022_metadata.json", 'w') as f:
            json.dump(metadata, f)

        # Also add summary.json for fallback
        summary = {
            "start": "2025-01-11 14:30:22",
            "num_accelerators": 8,
            "num_hosts": 1,
            "host_memory_GB": [256],
            "host_cpu_count": [64],
            "metric": {}
        }
        with open(result_dir / "summary.json", 'w') as f:
            json.dump(summary, f)

        # Create .hydra directory
        hydra_dir = result_dir / "dlio_config"
        hydra_dir.mkdir()
        config = {
            "workload": {
                "model": {"name": "unet3d"},
                "workflow": {"generate_data": False, "train": True, "checkpoint": False}
            }
        }
        with open(hydra_dir / "config.yaml", 'w') as f:
            import yaml
            yaml.dump(config, f)
        with open(hydra_dir / "overrides.yaml", 'w') as f:
            yaml.dump(["workload=unet3d_h100"], f)

        return result_dir

    def test_uses_metadata_when_complete(self, mock_logger, complete_metadata_dir):
        """ResultFilesExtractor uses metadata file when complete."""
        extractor = ResultFilesExtractor()
        result = extractor.extract(str(complete_metadata_dir), logger=mock_logger)

        assert isinstance(result, BenchmarkRunData)
        assert result.model == "unet3d"
        assert result.command == "run"
        assert result.num_processes == 8

    def test_falls_back_to_parser(self, mock_logger, incomplete_metadata_dir):
        """ResultFilesExtractor falls back to parser when metadata incomplete."""
        extractor = ResultFilesExtractor()
        result = extractor.extract(str(incomplete_metadata_dir), logger=mock_logger)

        assert isinstance(result, BenchmarkRunData)
        # Should have extracted from DLIO files
        assert result.benchmark_type == BENCHMARK_TYPES.training

    def test_accepts_custom_parser(self, mock_logger, tmp_path):
        """ResultFilesExtractor accepts custom result parser."""
        result_dir = tmp_path / "custom"
        result_dir.mkdir()

        # Create mock parser
        mock_parser = MagicMock()
        mock_parser.parse.return_value = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="custom_model",
            command="run",
            run_datetime="20250111_143022",
            num_processes=4,
            parameters={},
            override_parameters={}
        )

        extractor = ResultFilesExtractor(result_parser=mock_parser)
        result = extractor.extract(str(result_dir), logger=mock_logger)

        assert result.model == "custom_model"
        mock_parser.parse.assert_called_once()

    def test_is_complete_metadata_checks_required_fields(self):
        """_is_complete_metadata checks for required fields."""
        extractor = ResultFilesExtractor()

        complete = {
            'benchmark_type': 'training',
            'run_datetime': '20250111_143022',
            'num_processes': 8,
            'parameters': {}
        }
        assert extractor._is_complete_metadata(complete) is True

        incomplete = {
            'benchmark_type': 'training',
            'run_datetime': '20250111_143022'
            # Missing num_processes and parameters
        }
        assert extractor._is_complete_metadata(incomplete) is False

    def test_from_metadata_converts_benchmark_type(self, mock_logger):
        """_from_metadata correctly converts benchmark_type string to enum."""
        extractor = ResultFilesExtractor()

        metadata = {
            'benchmark_type': 'training',
            'model': 'unet3d',
            'command': 'run',
            'run_datetime': '20250111_143022',
            'num_processes': 8,
            'parameters': {}
        }

        result = extractor._from_metadata(metadata, '/path/to/results')

        assert result.benchmark_type == BENCHMARK_TYPES.training

    def test_from_metadata_handles_checkpointing_type(self, mock_logger):
        """_from_metadata handles checkpointing benchmark type."""
        extractor = ResultFilesExtractor()

        metadata = {
            'benchmark_type': 'checkpointing',
            'model': 'llama3-8b',
            'command': 'run',
            'run_datetime': '20250111_143022',
            'num_processes': 8,
            'parameters': {}
        }

        result = extractor._from_metadata(metadata, '/path/to/results')

        assert result.benchmark_type == BENCHMARK_TYPES.checkpointing

    def test_load_metadata_handles_no_metadata_file(self, mock_logger, tmp_path):
        """_load_metadata returns None when no metadata file exists."""
        empty_dir = tmp_path / "no_metadata"
        empty_dir.mkdir()

        extractor = ResultFilesExtractor()
        result = extractor._load_metadata(str(empty_dir), logger=mock_logger)

        assert result is None

    def test_load_metadata_handles_invalid_json(self, mock_logger, tmp_path):
        """_load_metadata returns None for invalid JSON."""
        result_dir = tmp_path / "invalid_json"
        result_dir.mkdir()

        # Write invalid JSON
        with open(result_dir / "test_metadata.json", 'w') as f:
            f.write("not valid json {")

        extractor = ResultFilesExtractor()
        result = extractor._load_metadata(str(result_dir), logger=mock_logger)

        assert result is None


class TestExtractorIntegration:
    """Integration tests using fixture data from tests/fixtures."""

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

    def test_extract_from_training_fixture(self, training_fixture_dir):
        """Extract data from training fixture directory."""
        extractor = ResultFilesExtractor()
        result = extractor.extract(str(training_fixture_dir))

        assert isinstance(result, BenchmarkRunData)
        # Verify extracted data matches fixture
        assert result.benchmark_type == BENCHMARK_TYPES.training
        assert result.model is not None

    def test_extract_from_checkpointing_fixture(self, checkpointing_fixture_dir):
        """Extract data from checkpointing fixture directory."""
        extractor = ResultFilesExtractor()
        result = extractor.extract(str(checkpointing_fixture_dir))

        assert isinstance(result, BenchmarkRunData)
        assert result.benchmark_type == BENCHMARK_TYPES.checkpointing
