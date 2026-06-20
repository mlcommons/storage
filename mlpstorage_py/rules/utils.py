"""
Utility functions for rules validation.

This module contains helper functions used by rules checkers and other
components for calculating requirements and generating output paths.
"""

import os
import re
import sys
from typing import Tuple, List, Optional

from mlpstorage_py.config import BENCHMARK_TYPES, DATETIME_STR
from mlpstorage_py.errors import ConfigurationError, ErrorCode

# Env-var names used by the Phase 2 CLI dispatch layer to source orgname/systemname (D-01, D-02).
# generate_output_location itself does NOT read these; the helper in
# mlpstorage_py/submission_checker/tools/code_image.py reads + validates them and threads
# the values through as keyword arguments. The names are exported here so the helper has a
# single source of truth for the env-var spelling.
MLPSTORAGE_ORGNAME_ENVVAR = "MLPSTORAGE_ORGNAME"
MLPSTORAGE_SYSTEMNAME_ENVVAR = "MLPSTORAGE_SYSTEMNAME"

# Each path segment appended to results_dir by generate_output_location must
# match this — POSIX-safe alphanumeric plus '.', '_', '-' — and must not be
# '.' or '..'. Blocks path-traversal ('../') and absolute-path resets ('/')
# at the trust boundary between args/env-var input and os.path.join, even
# for callers that bypass the CLI's argparse choices= validation.
_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_safe_path_component(name: str, value: str) -> None:
    """Raise ValueError if value is not safe as a single path segment.

    Caller handles None/empty upstream as a separate "missing required arg"
    failure mode; this helper assumes value is a non-empty string.
    """
    if value in (".", ".."):
        raise ValueError(
            f"{name}={value!r} is not a safe path component (reserved name)"
        )
    if not _SAFE_PATH_COMPONENT_RE.match(value):
        raise ValueError(
            f"{name}={value!r} is not a safe path component "
            f"(must match {_SAFE_PATH_COMPONENT_RE.pattern})"
        )


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

    # Calculate record length
    if 'record_length_bytes' in dataset_params:
        record_length_bytes = dataset_params['record_length_bytes']
    elif dataset_params.get('format') == 'parquet' and 'parquet' in dataset_params:
        # Calculate record length from parquet columns
        record_length_bytes = 0
        columns = dataset_params['parquet'].get('columns', [])
        for col in columns:
            dtype = col.get('dtype', 'float32')
            size = int(col.get('size', 1))
            
            if dtype == 'float64' or dtype == 'int64':
                record_length_bytes += size * 8
            elif dtype == 'uint8' or dtype == 'bool':
                record_length_bytes += size * 1
            else:
                # Default to float32/int32 (4 bytes)
                record_length_bytes += size * 4
    else:
        record_length_bytes = 0
        logger.warning("Could not determine record_length_bytes. Defaulting to 0.")

    file_size_bytes = dataset_params['num_samples_per_file'] * record_length_bytes

    if file_size_bytes > 0:
        min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    else:
        min_num_files_by_bytes = 0
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024}MiB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')

    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


def generate_output_location(
    benchmark,
    datetime_str=None,
    *,
    orgname: Optional[str] = None,
    systemname: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate a standardized output location for benchmark results.

    Output structure follows this pattern:

      CLOSED (args.mode == "closed"):
        <results_dir>/closed/<orgname>/<benchmark_name>/<model>/<command>/<datetime>/

      OPEN (args.mode == "open"):
        <results_dir>/open/<orgname>/results/<systemname>/<benchmark_name>/<model>/<command>/<datetime>/

      Legacy (args.mode not in {"closed", "open"}, or attribute missing —
      e.g. whatif, programmatic callers from tests):
        <results_dir>/<benchmark_name>/<model>/<command>/<datetime>/

    The per-``BENCHMARK_TYPES`` tail (training/checkpointing/vector_database/
    kv_cache) is unchanged below the new prefix.

    Args:
        benchmark: Benchmark instance.
        datetime_str: Optional datetime string for the run.
        orgname: Keyword-only. Submitter organization name; required when
            ``benchmark.args.mode`` is "closed" or "open". The CLI dispatch
            layer (Plan 02-02) reads ``MLPSTORAGE_ORGNAME`` from the
            environment, validates it per Rules.md §2.1.1, and threads the
            validated value through as this keyword argument. This function
            does NOT read ``os.environ`` — passing the value explicitly is a
            trust-contract requirement so programmatic callers (tests,
            future tooling) receive a typed ``ConfigurationError`` if they
            forget to thread it through, rather than a hidden ``KeyError``.
        systemname: Keyword-only. System name; required when
            ``benchmark.args.mode`` is "open". Same trust-contract semantics
            as ``orgname``; sourced from ``MLPSTORAGE_SYSTEMNAME`` by the
            dispatch layer.
        **kwargs: Additional benchmark-specific parameters (reserved).

    Returns:
        Full path to the output location.

    Raises:
        ValueError: If required parameters are missing (e.g. ``args.model``
            for training/checkpointing benchmarks).
        ConfigurationError: If ``benchmark.args.mode`` is "closed" or "open"
            but ``orgname`` (and, for "open", ``systemname``) was not threaded
            through by the caller. The ``parameter`` attribute identifies the
            missing kwarg; the ``suggestion`` field references the
            ``MLPSTORAGE_*`` env-var the dispatch layer must read.
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    output_location = benchmark.args.results_dir

    if hasattr(benchmark, "run_number"):
        run_number = benchmark.run_number
    else:
        run_number = 0

    # New D-03 prefix: insert {closed|open}/<orgname>[/results/<systemname>]/
    # before the legacy per-type chain. The values are explicit kwargs threaded
    # by the CLI dispatch layer (Plan 02-02); env-var reading is owned by that
    # helper, not this function (see module-level constants above for the
    # env-var-name source of truth).
    mode = getattr(benchmark.args, "mode", None)
    if mode in ("closed", "open"):
        if not orgname:
            raise ConfigurationError(
                "orgname is required when args.mode in {closed, open} but was "
                "not provided to generate_output_location",
                parameter="orgname",
                suggestion=(
                    f"The CLI dispatch layer should read the "
                    f"{MLPSTORAGE_ORGNAME_ENVVAR} environment variable and "
                    "thread the validated value through as the orgname "
                    "keyword. Programmatic callers must pass orgname= "
                    "explicitly."
                ),
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        _check_safe_path_component("orgname", orgname)
        output_location = os.path.join(output_location, mode, orgname)

        if mode == "open":
            if not systemname:
                raise ConfigurationError(
                    "systemname is required when args.mode == 'open' but was "
                    "not provided to generate_output_location",
                    parameter="systemname",
                    suggestion=(
                        f"The CLI dispatch layer should read the "
                        f"{MLPSTORAGE_SYSTEMNAME_ENVVAR} environment "
                        "variable and thread the validated value through "
                        "as the systemname keyword. Programmatic callers "
                        "must pass systemname= explicitly."
                    ),
                    code=ErrorCode.CONFIG_MISSING_REQUIRED,
                )
            _check_safe_path_component("systemname", systemname)
            output_location = os.path.join(output_location, "results", systemname)

    # datetime_str is built into every per-type path below; validate once here.
    _check_safe_path_component("datetime_str", datetime_str)

    # Handle different benchmark types
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        _check_safe_path_component("model", benchmark.args.model)
        _check_safe_path_component("command", benchmark.args.command)
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        # Results split by index_type because AISAQ is not comparable to
        # DISKANN/HNSW — they must live in separate on-disk trees so
        # submission validation and downstream tooling never collate them
        # (per Rules.md §2.1.27).
        engine = getattr(benchmark.args, "vdb_engine", None)
        if not engine:
            raise ValueError(
                "VectorDB engine is required for output location "
                "(set --vdb-engine on the CLI)."
            )
        vdb_index = (
            getattr(benchmark.args, "vdb_index", None)
            or getattr(benchmark.args, "index_type", None)
        )
        if not vdb_index:
            raise ValueError(
                "VectorDB index is required for output location "
                "(set --vdb-index on the CLI)."
            )

        _check_safe_path_component("vdb_engine", engine)
        _check_safe_path_component("vdb_index", vdb_index)
        _check_safe_path_component("command", benchmark.args.command)
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, engine)
        output_location = os.path.join(output_location, vdb_index)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.kv_cache:
        model = getattr(benchmark.args, "model", None)
        if not model:
            raise ValueError(
                "Model is required for kv_cache output location: set "
                "args.model before calling generate_output_location "
                "(KVCacheBenchmark.__init__ defaults this from KVCACHE_MODEL_DEFAULT)."
            )
        _check_safe_path_component("model", model)
        _check_safe_path_component("command", benchmark.args.command)
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for checkpointing benchmark output location")

        _check_safe_path_component("model", benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, datetime_str)

    else:
        print(f'The given benchmark is not supported by mlpstorage_py.rules.generate_output_location()')
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
    from mlpstorage_py.rules.models import BenchmarkRun

    runs = []

    if not os.path.exists(results_dir):
        if logger:
            logger.warning(f"Results directory not found: {results_dir}")
        return runs

    # Walk the directory tree looking for run directories. followlinks=True
    # lets users symlink previously-completed run directories into a fresh
    # results-dir to accumulate them — a common workflow when stitching
    # together results from multiple machines or earlier runs.
    for root, dirs, files in os.walk(results_dir, followlinks=True):
        # Check if this directory contains a summary.json (DLIO run) or metadata file
        has_summary = 'summary.json' in files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]
        has_metadata = len(metadata_files) == 1

        if len(metadata_files) > 1:
            if logger:
                logger.warning(f"Skipping {root}: multiple metadata files found ({len(metadata_files)})")
            continue

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
