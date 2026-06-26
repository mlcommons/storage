"""
Tests for calculation functions in mlpstorage.rules module.

Tests cover:
- calculate_training_data_size: dataset size calculation based on memory and steps
- generate_output_location: output path generation for benchmark results
- get_runs_files: directory traversal for benchmark results
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mlpstorage_py.config import BENCHMARK_TYPES
from mlpstorage_py.rules import (
    calculate_training_data_size,
    generate_output_location,
    get_runs_files,
    ClusterInformation,
    HostInfo,
    HostMemoryInfo,
    BenchmarkRun,
)


class TestCalculateTrainingDataSize:
    """Tests for calculate_training_data_size function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger with all custom log methods."""
        logger = MagicMock()
        logger.ridiculous = MagicMock()
        logger.result = MagicMock()
        return logger

    @pytest.fixture
    def sample_dataset_params(self):
        """Sample dataset parameters."""
        return {
            'num_samples_per_file': 1,  # Simplified: 1 sample per file
            'record_length_bytes': 131072,  # 128 KB per sample
        }

    @pytest.fixture
    def sample_reader_params(self):
        """Sample reader parameters."""
        return {
            'batch_size': 7,  # Common batch size for unet3d
        }

    @pytest.fixture
    def sample_cluster_info(self, mock_logger):
        """Create sample cluster information."""
        # 256 GB total memory (2 hosts with 128 GB each)
        host1_memory = HostMemoryInfo.from_total_mem_int(137438953472)  # 128 GB
        host2_memory = HostMemoryInfo.from_total_mem_int(137438953472)  # 128 GB
        host1 = HostInfo(hostname='host1', memory=host1_memory)
        host2 = HostInfo(hostname='host2', memory=host2_memory)
        return ClusterInformation([host1, host2], mock_logger)

    def test_calculates_with_cluster_info(self, mock_logger, sample_cluster_info,
                                          sample_dataset_params, sample_reader_params):
        """calculate_training_data_size uses cluster_information when args is None."""
        num_files, num_subfolders, total_bytes = calculate_training_data_size(
            args=None,
            cluster_information=sample_cluster_info,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger,
            num_processes=8
        )

        assert num_files > 0
        assert isinstance(num_files, int)
        assert total_bytes > 0

    def test_calculates_with_args(self, mock_logger, sample_dataset_params, sample_reader_params):
        """calculate_training_data_size uses args when provided."""
        mock_args = MagicMock()
        mock_args.client_host_memory_in_gb = 128  # 128 GB per host
        mock_args.num_client_hosts = 2  # 2 hosts
        mock_args.num_processes = 8

        num_files, num_subfolders, total_bytes = calculate_training_data_size(
            args=mock_args,
            cluster_information=None,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger
        )

        assert num_files > 0
        assert isinstance(num_files, int)

    def test_memory_rule_takes_precedence(self, mock_logger, sample_dataset_params, sample_reader_params):
        """When memory requirement exceeds step requirement, memory rule wins."""
        # Large memory, few processes = memory rule should dominate
        mock_args = MagicMock()
        mock_args.client_host_memory_in_gb = 512  # Large memory
        mock_args.num_client_hosts = 4
        mock_args.num_processes = 2  # Few processes

        num_files, _, _ = calculate_training_data_size(
            args=mock_args,
            cluster_information=None,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger
        )

        # With 2 TB memory and 5x requirement, need at least 10 TB / file_size files
        # Check that result is driven by memory (should be large)
        assert num_files > 1000

    def test_steps_rule_takes_precedence(self, mock_logger, sample_dataset_params, sample_reader_params):
        """When step requirement exceeds memory requirement, step rule wins."""
        # Small memory, many processes = step rule should dominate
        mock_args = MagicMock()
        mock_args.client_host_memory_in_gb = 8  # Small memory
        mock_args.num_client_hosts = 1
        mock_args.num_processes = 64  # Many processes

        num_files, _, _ = calculate_training_data_size(
            args=mock_args,
            cluster_information=None,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger
        )

        # 500 steps * 64 processes * 7 batch_size = 224,000 samples needed
        # With 1 sample per file, need at least 224,000 files
        assert num_files >= 224000

    def test_returns_tuple_of_three(self, mock_logger, sample_dataset_params, sample_reader_params):
        """calculate_training_data_size returns tuple of (num_files, num_subfolders, total_bytes)."""
        mock_args = MagicMock()
        mock_args.client_host_memory_in_gb = 64
        mock_args.num_client_hosts = 1
        mock_args.num_processes = 8

        result = calculate_training_data_size(
            args=mock_args,
            cluster_information=None,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        num_files, num_subfolders, total_bytes = result
        assert isinstance(num_files, int)
        assert isinstance(num_subfolders, int)
        assert isinstance(total_bytes, int)

    def test_subfolders_defaults_to_zero(self, mock_logger, sample_dataset_params, sample_reader_params):
        """num_subfolders defaults to 0."""
        mock_args = MagicMock()
        mock_args.client_host_memory_in_gb = 64
        mock_args.num_client_hosts = 1
        mock_args.num_processes = 8

        _, num_subfolders, _ = calculate_training_data_size(
            args=mock_args,
            cluster_information=None,
            dataset_params=sample_dataset_params,
            reader_params=sample_reader_params,
            logger=mock_logger
        )

        assert num_subfolders == 0

    def test_raises_without_args_or_cluster_info(self, mock_logger, sample_dataset_params, sample_reader_params):
        """Raises ValueError when neither args nor cluster_information provides memory."""
        mock_args = MagicMock()
        # No client_host_memory_in_gb or num_client_hosts
        mock_args.client_host_memory_in_gb = None
        mock_args.num_client_hosts = None
        mock_args.clienthost_host_memory_in_gb = None

        with pytest.raises(ValueError, match="Either args or cluster_information is required"):
            calculate_training_data_size(
                args=mock_args,
                cluster_information=None,
                dataset_params=sample_dataset_params,
                reader_params=sample_reader_params,
                logger=mock_logger
            )

    def test_raises_value_error_when_both_args_and_cluster_info_are_none(
            self, mock_logger, sample_dataset_params, sample_reader_params):
        """Issue #503 bug 1: with args=None and cluster_information=None the
        function must raise ValueError instead of AttributeError. This is the
        reportgen-loaded-from-disk path; the caller catches and skips."""
        with pytest.raises(ValueError, match="cluster_information"):
            calculate_training_data_size(
                args=None,
                cluster_information=None,
                dataset_params=sample_dataset_params,
                reader_params=sample_reader_params,
                logger=mock_logger,
            )


class TestGenerateOutputLocation:
    """Tests for generate_output_location function.

    After the Phase 1 / Plan 01-03 rewrite, generate_output_location emits
    the Rules.md §2.1-shaped canonical layout:

        <results-dir>/<mode>/<orgname>/results/<systemname>/<benchmark>/<model>/<command>/<datetime>/

    Checkpointing omits the <command> segment to preserve the pre-refactor
    behavior. The function reads args.{mode, orgname, systemname} purely;
    upstream orgname resolution lives in the Slice 4 main gate.
    """

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_training_benchmark_output_location(self, mock_logger):
        """generate_output_location creates correct canonical path for training benchmarks."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        # Canonical layout (LAY-05): <rd>/<mode>/<org>/results/<sys>/<bench>/<model>/<cmd>/<dt>/
        assert '/results/closed/Acme/results/sys-v1/training/unet3d/run/20250111_143022' == result

    def test_checkpointing_benchmark_output_location(self, mock_logger):
        """generate_output_location creates correct canonical path for checkpointing benchmarks.

        Checkpointing intentionally omits the <command> segment — preserves
        the pre-refactor shape so existing fixtures/checkers stay aligned.
        """
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.checkpointing
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.model = 'llama3-8b'

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        assert '/results/closed/Acme/results/sys-v1/checkpointing/llama3-8b/20250111_143022' == result

    def test_raises_for_training_without_model(self, mock_logger):
        """generate_output_location raises ConfigurationError for training without model.

        Post WR-07: missing-required errors now raise the typed
        ``ConfigurationError`` so the top-level ``main()`` handler can
        surface them uniformly. Previously this was a bare ``ValueError``,
        which bypassed the typed-error handler.
        """
        from mlpstorage_py.errors import ConfigurationError
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        del mock_benchmark.args.model  # Remove model attribute

        with pytest.raises(ConfigurationError, match="Model name is required"):
            generate_output_location(mock_benchmark, datetime_str='20250111_143022')

    def test_raises_for_checkpointing_without_model(self, mock_logger):
        """generate_output_location raises ConfigurationError for checkpointing without model.

        Post WR-07: see ``test_raises_for_training_without_model``.
        """
        from mlpstorage_py.errors import ConfigurationError
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.checkpointing
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        del mock_benchmark.args.model  # Remove model attribute

        with pytest.raises(ConfigurationError, match="Model name is required"):
            generate_output_location(mock_benchmark, datetime_str='20250111_143022')

    def test_vectordb_benchmark_output_location(self, mock_logger):
        """generate_output_location creates correct canonical path for vectordb benchmarks."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.vector_database
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'open'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.vdb_engine = 'milvus'
        mock_benchmark.args.vdb_index = 'diskann'
        mock_benchmark.args.command = 'run'

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        assert '/results/open/Acme/results/sys-v1/vector_database/milvus/diskann/run/20250111_143022' == result

    def test_kvcache_benchmark_output_location(self, mock_logger):
        """generate_output_location creates correct canonical path for kv_cache benchmarks."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.kv_cache
        mock_benchmark.args.results_dir = '/results'
        mock_benchmark.args.mode = 'whatif'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.model = 'llama3.1-8b'
        mock_benchmark.args.command = 'run'

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        assert '/results/whatif/Acme/results/sys-v1/kv_cache/llama3.1-8b/run/20250111_143022' == result

    @pytest.mark.parametrize('mode', ['closed', 'open', 'whatif'])
    def test_canonical_prefix_training(self, mock_logger, mode):
        """Canonical-prefix smoke (training): output starts with <rd>/<mode>/<org>/results/<sys>/."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.results_dir = '/r'
        mock_benchmark.args.mode = mode
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        assert result.startswith(f'/r/{mode}/Acme/results/sys-v1/'), result

    @pytest.mark.parametrize('benchmark_type, model_or_engine_field, model_or_engine, expected_segment', [
        (BENCHMARK_TYPES.training, 'model', 'unet3d', 'training/unet3d/run/'),
        (BENCHMARK_TYPES.vector_database, 'vdb_engine', 'milvus', 'vector_database/milvus/diskann/run/'),
        (BENCHMARK_TYPES.kv_cache, 'model', 'llama3.1-8b', 'kv_cache/llama3.1-8b/run/'),
    ])
    def test_canonical_prefix_all_benchmark_types(
        self, mock_logger, benchmark_type, model_or_engine_field, model_or_engine,
        expected_segment,
    ):
        """Canonical-prefix smoke across benchmark types (training, vectordb, kvcache)."""
        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = benchmark_type
        mock_benchmark.args.results_dir = '/r'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.command = 'run'
        # vector_database also requires --vdb-index per Rules.md §2.1.27 split
        # (AISAQ vs DISKANN/HNSW must live in separate trees). Harmless for
        # other benchmark types since they don't read this attribute.
        mock_benchmark.args.vdb_index = 'diskann'
        setattr(mock_benchmark.args, model_or_engine_field, model_or_engine)

        result = generate_output_location(mock_benchmark, datetime_str='20250111_143022')

        assert result.startswith('/r/closed/Acme/results/sys-v1/'), result
        assert expected_segment in result, result

    def test_generate_output_location_empty_systemname_raises(self, mock_logger):
        """T-1-02: empty args.systemname must raise ConfigurationError, not produce //results//."""
        from mlpstorage_py.errors import ConfigurationError

        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.results_dir = '/r'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = 'Acme'
        mock_benchmark.args.systemname = ''  # explicitly empty (post-resolution miss)
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'

        with pytest.raises(ConfigurationError, match=r'systemname'):
            generate_output_location(mock_benchmark, datetime_str='20250111_143022')

    def test_generate_output_location_empty_orgname_raises(self, mock_logger):
        """Pitfall 1 defense-in-depth: empty args.orgname must raise ConfigurationError.

        The user-facing actionable error comes from main._main_impl()'s
        orgname-resolution gate (Slice 4). This raise is purely defensive
        — it guards against a code path that bypasses the gate.
        """
        from mlpstorage_py.errors import ConfigurationError

        mock_benchmark = MagicMock()
        mock_benchmark.BENCHMARK_TYPE = BENCHMARK_TYPES.training
        mock_benchmark.args.results_dir = '/r'
        mock_benchmark.args.mode = 'closed'
        mock_benchmark.args.orgname = ''  # sentinel not resolved
        mock_benchmark.args.systemname = 'sys-v1'
        mock_benchmark.args.model = 'unet3d'
        mock_benchmark.args.command = 'run'

        with pytest.raises(ConfigurationError, match=r'orgname'):
            generate_output_location(mock_benchmark, datetime_str='20250111_143022')


class TestGetRunsFiles:
    """Tests for get_runs_files function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger with custom methods."""
        logger = MagicMock()
        logger.ridiculous = MagicMock()
        logger.warning = MagicMock()
        logger.debug = MagicMock()
        logger.error = MagicMock()
        return logger

    @pytest.fixture
    def sample_results_dir(self, tmp_path):
        """Create a sample results directory structure."""
        # Create training run directory
        training_dir = tmp_path / "training" / "unet3d" / "run" / "20250111_143022"
        training_dir.mkdir(parents=True)

        # Create metadata file
        metadata = {
            "benchmark_type": "training",
            "model": "unet3d",
            "command": "run",
            "run_datetime": "20250111_143022",
            "num_processes": 8,
            "parameters": {"dataset": {"num_files_train": 400}},
            "override_parameters": {}
        }
        with open(training_dir / "training_20250111_143022_metadata.json", 'w') as f:
            json.dump(metadata, f)

        # Create summary.json
        summary = {
            "start": "2025-01-11 14:30:22",
            "num_accelerators": 8,
            "num_hosts": 1,
            "host_memory_GB": [256],
            "host_cpu_count": [64],
            "metric": {}
        }
        with open(training_dir / "summary.json", 'w') as f:
            json.dump(summary, f)

        # Create .hydra directory
        hydra_dir = training_dir / ".hydra"
        hydra_dir.mkdir()
        with open(hydra_dir / "config.yaml", 'w') as f:
            import yaml
            yaml.dump({
                "workload": {
                    "model": {"name": "unet3d"},
                    "workflow": {"train": True, "generate_data": False, "checkpoint": False}
                }
            }, f)
        with open(hydra_dir / "overrides.yaml", 'w') as f:
            yaml.dump(["workload=unet3d_h100"], f)

        return tmp_path

    def test_finds_benchmark_runs(self, mock_logger, sample_results_dir):
        """get_runs_files finds benchmark runs in results directory."""
        runs = get_runs_files(str(sample_results_dir), logger=mock_logger)

        assert len(runs) == 1
        assert isinstance(runs[0], BenchmarkRun)

    def test_returns_empty_for_nonexistent_dir(self, mock_logger, tmp_path):
        """get_runs_files returns empty list for non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        runs = get_runs_files(str(nonexistent), logger=mock_logger)

        assert runs == []
        mock_logger.warning.assert_called()

    def test_skips_dirs_without_metadata(self, mock_logger, tmp_path):
        """get_runs_files skips directories without metadata files."""
        # Create directory without metadata
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()

        runs = get_runs_files(str(tmp_path), logger=mock_logger)

        assert len(runs) == 0

    def test_skips_dirs_without_summary(self, mock_logger, tmp_path):
        """get_runs_files skips directories without summary.json."""
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        # Create metadata but no summary
        metadata = {"benchmark_type": "training", "run_datetime": "20250111"}
        with open(run_dir / "training_20250111_metadata.json", 'w') as f:
            json.dump(metadata, f)

        runs = get_runs_files(str(tmp_path), logger=mock_logger)

        assert len(runs) == 0

    def test_skips_dirs_with_multiple_metadata(self, mock_logger, tmp_path):
        """get_runs_files skips directories with multiple metadata files."""
        run_dir = tmp_path / "multiple_metadata"
        run_dir.mkdir()

        # Create multiple metadata files
        with open(run_dir / "training_1_metadata.json", 'w') as f:
            json.dump({}, f)
        with open(run_dir / "training_2_metadata.json", 'w') as f:
            json.dump({}, f)
        with open(run_dir / "summary.json", 'w') as f:
            json.dump({}, f)

        runs = get_runs_files(str(tmp_path), logger=mock_logger)

        assert len(runs) == 0
        mock_logger.warning.assert_called()

    def test_handles_parse_errors(self, mock_logger, tmp_path):
        """get_runs_files handles errors when parsing run directories."""
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()

        # Create invalid metadata
        with open(run_dir / "bad_metadata.json", 'w') as f:
            f.write("not valid json {")
        with open(run_dir / "summary.json", 'w') as f:
            json.dump({"num_accelerators": 8}, f)

        runs = get_runs_files(str(tmp_path), logger=mock_logger)

        # Should not crash, but may not find valid runs
        assert isinstance(runs, list)
