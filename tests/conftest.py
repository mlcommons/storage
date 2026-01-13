"""
Shared pytest fixtures for mlpstorage tests.

These fixtures provide mock data, loggers, and test utilities that can be
used across all test modules without requiring DLIO to be installed.
"""

import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_results_dir(fixtures_dir) -> Path:
    """Return path to sample results fixtures."""
    return fixtures_dir / "sample_results"


# =============================================================================
# Logger Fixtures
# =============================================================================

@pytest.fixture
def mock_logger():
    """
    Create a mock logger that captures all log calls.

    Usage:
        def test_something(mock_logger):
            some_function(logger=mock_logger)
            mock_logger.info.assert_called_with("expected message")
    """
    logger = MagicMock()
    # Add all log levels as mock methods
    for level in ['debug', 'info', 'warning', 'error', 'critical',
                  'status', 'verbose', 'verboser', 'ridiculous', 'result']:
        setattr(logger, level, MagicMock())
    return logger


@pytest.fixture
def capturing_logger():
    """
    Create a logger that captures messages to a list.

    Usage:
        def test_something(capturing_logger):
            logger, messages = capturing_logger
            some_function(logger=logger)
            assert "expected" in messages['info'][0]
    """
    messages = {
        'debug': [], 'info': [], 'warning': [], 'error': [],
        'critical': [], 'status': [], 'verbose': [], 'verboser': [],
        'ridiculous': [], 'result': []
    }

    logger = MagicMock()

    def make_capture(level):
        def capture(msg, *args, **kwargs):
            messages[level].append(msg)
        return capture

    for level in messages:
        setattr(logger, level, make_capture(level))

    return logger, messages


# =============================================================================
# Args Fixtures (Namespace objects for CLI simulation)
# =============================================================================

@pytest.fixture
def base_args() -> Namespace:
    """Base args that all commands share."""
    return Namespace(
        debug=False,
        verbose=False,
        what_if=False,
        allow_invalid_params=False,
        results_dir='/tmp/test_results',
        loops=1,
    )


@pytest.fixture
def training_datasize_args(base_args) -> Namespace:
    """Args for training datasize command."""
    base_args.command = 'datasize'
    base_args.model = 'unet3d'
    base_args.accelerator_type = 'h100'
    base_args.num_client_hosts = 2
    base_args.client_host_memory_in_gb = 256
    base_args.max_accelerators = 16
    base_args.num_processes = 16
    base_args.hosts = ['127.0.0.1', '127.0.0.2']
    base_args.data_dir = '/data/unet3d'
    base_args.exec_type = 'mpi'
    base_args.mpi_bin = 'mpirun'
    base_args.oversubscribe = False
    base_args.allow_run_as_root = False
    base_args.params = None
    base_args.dlio_bin_path = None
    base_args.mpi_params = None
    return base_args


@pytest.fixture
def training_run_args(training_datasize_args) -> Namespace:
    """Args for training run command."""
    training_datasize_args.command = 'run'
    training_datasize_args.num_accelerators = 8
    training_datasize_args.checkpoint_folder = '/data/checkpoints'
    return training_datasize_args


@pytest.fixture
def checkpointing_args(base_args) -> Namespace:
    """Args for checkpointing command."""
    base_args.command = 'run'
    base_args.model = 'llama3-8b'
    base_args.num_processes = 8
    base_args.hosts = ['127.0.0.1']
    base_args.client_host_memory_in_gb = 512
    base_args.checkpoint_folder = '/data/checkpoints'
    base_args.num_checkpoints_read = 1
    base_args.num_checkpoints_write = 1
    base_args.exec_type = 'mpi'
    base_args.mpi_bin = 'mpirun'
    base_args.oversubscribe = False
    base_args.allow_run_as_root = True
    base_args.params = None
    base_args.dlio_bin_path = None
    base_args.mpi_params = None
    return base_args


# =============================================================================
# BenchmarkRunData Fixtures
# =============================================================================

@pytest.fixture
def sample_training_parameters() -> Dict[str, Any]:
    """Sample training parameters as would be loaded from YAML config."""
    return {
        'model': {'name': 'unet3d'},
        'dataset': {
            'num_files_train': 42000,
            'num_subfolders_train': 0,
            'data_folder': '/data/unet3d',
            'format': 'npz',
            'num_samples_per_file': 1,
        },
        'reader': {
            'read_threads': 8,
            'computation_threads': 1,
            'prefetch_size': 2,
            'transfer_size': 262144,
        },
        'workflow': {
            'generate_data': False,
            'train': True,
            'checkpoint': True,
        },
        'checkpoint': {
            'checkpoint_folder': '/data/checkpoints',
        },
    }


@pytest.fixture
def sample_checkpointing_parameters() -> Dict[str, Any]:
    """Sample checkpointing parameters."""
    return {
        'model': {'name': 'llama3_8b'},
        'checkpoint': {
            'checkpoint_folder': '/data/checkpoints',
            'num_checkpoints_read': 1,
            'num_checkpoints_write': 1,
            'mode': 'default',
        },
        'workflow': {
            'generate_data': False,
            'train': False,
            'checkpoint': True,
        },
    }


@pytest.fixture
def sample_cluster_info(mock_logger):
    """Create a sample ClusterInformation object."""
    from mlpstorage.rules import ClusterInformation, HostInfo, HostMemoryInfo

    host_info_list = [
        HostInfo(
            hostname='host1',
            cpu=None,
            memory=HostMemoryInfo.from_total_mem_int(256 * 1024**3)  # 256 GB
        ),
        HostInfo(
            hostname='host2',
            cpu=None,
            memory=HostMemoryInfo.from_total_mem_int(256 * 1024**3)  # 256 GB
        ),
    ]
    return ClusterInformation(host_info_list=host_info_list, logger=mock_logger)


@pytest.fixture
def sample_benchmark_run_data(sample_training_parameters, sample_cluster_info):
    """Create a sample BenchmarkRunData for testing."""
    from mlpstorage.rules import BenchmarkRunData

    return BenchmarkRunData(
        benchmark_type=BENCHMARK_TYPES.training,
        model='unet3d',
        command='run',
        run_datetime='20250111_143022',
        num_processes=8,
        parameters=sample_training_parameters,
        override_parameters={'dataset.num_files_train': '42000'},
        system_info=sample_cluster_info,
        metrics={'train_au_percentage': [95.2, 94.8, 95.1]},
        result_dir='/tmp/results/training/unet3d/run/20250111_143022',
        accelerator='h100',
    )


# =============================================================================
# Temporary Result Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_result_dir(tmp_path, sample_training_parameters):
    """
    Create a temporary result directory with mock DLIO output files.

    This fixture creates a realistic result directory structure that can be
    parsed by DLIOResultParser without needing actual DLIO execution.
    """
    result_dir = tmp_path / "training" / "unet3d" / "run" / "20250111_143022"
    result_dir.mkdir(parents=True)

    # Create summary.json (DLIO output)
    summary = {
        "start": "2025-01-11 14:30:22",
        "end": "2025-01-11 14:35:45",
        "num_accelerators": 8,
        "num_hosts": 2,
        "host_memory_GB": [256, 256],
        "host_cpu_count": [64, 64],
        "workload": "unet3d_h100",
        "metric": {
            "train_au_percentage": [95.2, 94.8, 95.1],
            "train_throughput_samples_per_second": [1250.5, 1248.2, 1251.0],
        }
    }
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create metadata file
    metadata = {
        "benchmark_type": "training",
        "model": "unet3d",
        "command": "run",
        "run_datetime": "20250111_143022",
        "num_processes": 8,
        "parameters": sample_training_parameters,
        "override_parameters": {"dataset.num_files_train": "42000"},
        "accelerator": "h100",
    }
    with open(result_dir / "training_20250111_143022_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create .hydra directory with configs
    hydra_dir = result_dir / ".hydra"
    hydra_dir.mkdir()

    # config.yaml
    config_yaml = {
        "workload": sample_training_parameters
    }
    with open(hydra_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config_yaml, f)

    # overrides.yaml
    overrides = [
        "workload=unet3d_h100",
        "++workload.dataset.num_files_train=42000",
        "++workload.dataset.data_folder=/data/unet3d",
    ]
    with open(hydra_dir / "overrides.yaml", "w") as f:
        import yaml
        yaml.dump(overrides, f)

    return result_dir


@pytest.fixture
def temp_checkpointing_result_dir(tmp_path, sample_checkpointing_parameters):
    """Create a temporary result directory for checkpointing benchmark."""
    result_dir = tmp_path / "checkpointing" / "llama3-8b" / "20250111_150000"
    result_dir.mkdir(parents=True)

    summary = {
        "start": "2025-01-11 15:00:00",
        "end": "2025-01-11 15:05:30",
        "num_accelerators": 8,
        "num_hosts": 1,
        "host_memory_GB": [512],
        "host_cpu_count": [64],
        "workload": "llama3_8b",
        "metric": {
            "checkpoint_write_throughput_GB_per_second": [45.2, 44.8],
            "checkpoint_read_throughput_GB_per_second": [52.1, 51.8],
        }
    }
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    metadata = {
        "benchmark_type": "checkpointing",
        "model": "llama3-8b",
        "command": "run",
        "run_datetime": "20250111_150000",
        "num_processes": 8,
        "parameters": sample_checkpointing_parameters,
        "override_parameters": {},
    }
    with open(result_dir / "checkpointing_20250111_150000_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    hydra_dir = result_dir / ".hydra"
    hydra_dir.mkdir()

    config_yaml = {"workload": sample_checkpointing_parameters}
    with open(hydra_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config_yaml, f)

    overrides = ["workload=llama3_8b"]
    with open(hydra_dir / "overrides.yaml", "w") as f:
        import yaml
        yaml.dump(overrides, f)

    return result_dir


# =============================================================================
# Mock Benchmark Instance Fixtures
# =============================================================================

@pytest.fixture
def mock_benchmark_instance(training_run_args, sample_training_parameters, sample_cluster_info):
    """
    Create a mock Benchmark instance for testing extractors.

    This mock has all the attributes that BenchmarkInstanceExtractor expects.
    """
    mock = MagicMock()
    mock.BENCHMARK_TYPE = BENCHMARK_TYPES.training
    mock.args = training_run_args
    mock.run_datetime = '20250111_143022'
    mock.combined_params = sample_training_parameters
    mock.params_dict = {'dataset.num_files_train': '42000'}
    mock.cluster_information = sample_cluster_info
    mock.run_result_output = '/tmp/results/training/unet3d/run/20250111_143022'
    return mock


# =============================================================================
# Environment Variable Fixtures
# =============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """
    Remove mlpstorage-related environment variables.

    Usage:
        def test_check_env_default(clean_env):
            # MLPS_DEBUG etc. are guaranteed to be unset
            result = check_env('MLPS_DEBUG', False)
            assert result is False
    """
    env_vars = ['MLPS_DEBUG', 'MLPS_VERBOSE', 'MLPS_WHAT_IF']
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch
