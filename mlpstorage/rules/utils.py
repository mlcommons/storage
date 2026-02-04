"""
Utility functions for rules validation.

This module contains helper functions used by rules checkers and other
components for calculating requirements and generating output paths.
"""

import os
import sys
from typing import Tuple, List, Optional

from mlpstorage.config import BENCHMARK_TYPES, DATETIME_STR


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger,
                                 num_processes=None) -> Tuple[int, int, int]:
    """
    Calculate the required training data size for closed submission.

    Requirements:
      - Dataset needs to be 5x the amount of total memory
      - Training needs to do at least 500 steps per epoch

    Memory Ratio:
      - Collect "Total Memory" from /proc/meminfo on each host
      - Sum it up
      - Multiply by 5
      - Divide by sample size
      - Divide by batch size

    500 steps:
      - 500 steps per epoch
      - Multiply by max number of processes
      - Multiply by batch size

    Args:
        args: Command-line arguments (optional, can be None).
        cluster_information: ClusterInformation instance with system info.
        dataset_params: Dataset parameters from benchmark config.
        reader_params: Reader parameters from benchmark config.
        logger: Logger instance.
        num_processes: Number of processes (optional).

    Returns:
        Tuple of (required_file_count, required_subfolders_count, total_disk_bytes)
    """
    required_file_count = 1
    required_subfolders_count = 0

    # Find the amount of memory in the cluster via args or measurements
    if not args:
        total_mem_bytes = cluster_information.total_memory_bytes
    elif hasattr(args, 'client_host_memory_in_gb') and args.client_host_memory_in_gb and \
         hasattr(args, 'num_client_hosts') and args.num_client_hosts:
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    elif hasattr(args, 'clienthost_host_memory_in_gb') and args.clienthost_host_memory_in_gb and \
         not (hasattr(args, 'num_client_hosts') and args.num_client_hosts):
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    else:
        raise ValueError('Either args or cluster_information is required')

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes
    file_size_bytes = dataset_params['num_samples_per_file'] * dataset_params['record_length_bytes']

    min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024} MB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')

    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


def generate_output_location(benchmark, datetime_str=None, **kwargs) -> str:
    """
    Generate a standardized output location for benchmark results.

    Output structure follows this pattern:
    RESULTS_DIR:
        <benchmark_name>:
            <model>:
                <command>:
                        <datetime>:
                            run_<run_number> (Optional)

    Args:
        benchmark: Benchmark instance.
        datetime_str: Optional datetime string for the run.
        **kwargs: Additional benchmark-specific parameters.

    Returns:
        Full path to the output location.

    Raises:
        ValueError: If required parameters are missing.
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    output_location = benchmark.args.results_dir

    if hasattr(benchmark, "run_number"):
        run_number = benchmark.run_number
    else:
        run_number = 0

    # Handle different benchmark types
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for checkpointing benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, datetime_str)

    else:
        print(f'The given benchmark is not supported by mlpstorage.rules.generate_output_location()')
        sys.exit(1)

    return output_location


def get_runs_files(results_dir: str, logger=None) -> List:
    """
    Find all benchmark run directories in a results directory.

    Args:
        results_dir: Path to the results directory.
        logger: Optional logger instance.

    Returns:
        List of BenchmarkRun instances.
    """
    from mlpstorage.rules.models import BenchmarkRun

    runs = []

    if not os.path.exists(results_dir):
        if logger:
            logger.error(f"Results directory not found: {results_dir}")
        return runs

    # Walk the directory tree looking for run directories
    for root, dirs, files in os.walk(results_dir):
        # Check if this directory contains a summary.json (DLIO run) or metadata file
        has_summary = 'summary.json' in files
        has_metadata = any(f.endswith('_metadata.json') for f in files)

        if has_summary or has_metadata:
            try:
                run = BenchmarkRun.from_result_dir(root, logger)
                runs.append(run)
                if logger:
                    logger.debug(f"Found run: {run.run_id}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load run from {root}: {e}")

    return runs
