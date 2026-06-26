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

# Env-var names used by the CLI dispatch layer to source orgname/systemname.
# generate_output_location itself does NOT read these; values are threaded in
# via benchmark.args (populated upstream by main._main_impl()'s sentinel-
# resolution gate). The names are exported here as a single source of truth
# for the env-var spelling.
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
        if cluster_information is None:
            # Loaded-from-disk runs (reportgen path) may lack the live
            # ClusterInformation that an in-process run collects. Without
            # total_memory_bytes the 5×memory rule cannot be enforced — raise
            # a clear error so the caller (check_num_files_train) can turn it
            # into a non-fatal "skipped" notice rather than crashing the entire
            # verification with an AttributeError. (#503)
            raise ValueError(
                "calculate_training_data_size requires either args or a "
                "non-None cluster_information; both were missing (typical when "
                "loading benchmark runs from on-disk metadata that lacks "
                "cluster_information)"
            )
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
    Generate the canonical Rules.md §2.1-shaped output path for benchmark results.

    Canonical shape (LAY-05, Phase 1 Plan 01-03):

        <results-dir>/<mode>/<orgname>/results/<systemname>/<benchmark>/<model>/<command>/<datetime>/

    Vector-database results include the index_type between engine and command
    (closed/open results for AISAQ vs DISKANN/HNSW must live in separate
    trees per Rules.md §2.1.27):

        <results-dir>/<mode>/<orgname>/results/<systemname>/vector_database/<engine>/<index>/<command>/<datetime>/

    Checkpointing intentionally omits the <command> segment to preserve the
    pre-refactor layout of checkpointing runs:

        <results-dir>/<mode>/<orgname>/results/<systemname>/checkpointing/<model>/<datetime>/

    This function is PURE with respect to args.{mode, orgname, systemname} —
    it does NOT resolve orgname from the sentinel or read MLPERF_SYSTEMNAME
    here. Per RESEARCH.md Pitfall 1, orgname resolution lives upstream in
    main._main_impl()'s sentinel-resolution gate (Slice 4); the universal
    --systemname plumbing (Slice 3 / this plan) populates args.systemname.

    Every path segment appended to results_dir is validated via
    _check_safe_path_component() to block path-traversal ('../') and
    absolute-path resets ('/') at the trust boundary, even for callers that
    bypass the CLI's argparse choices= validation.

    Args:
        benchmark: Benchmark instance. Expected attributes:
            - benchmark.BENCHMARK_TYPE — one of BENCHMARK_TYPES enum values.
            - benchmark.args.results_dir, args.mode, args.orgname, args.systemname.
            - benchmark.args.{model | vdb_engine[, vdb_index]}, args.command
              (per BENCHMARK_TYPE).
        datetime_str: Optional datetime string for the run; defaults to
            mlpstorage_py.config.DATETIME_STR.
        **kwargs: Reserved for forward compatibility; currently unused.

    Returns:
        Full path to the output location, no trailing slash.

    Raises:
        ConfigurationError: If args.systemname is empty (T-1-02 mitigation —
            empty post-resolution systemname would silently produce
            "<rd>/closed/Acme/results//training/..." which subsequent
            os.makedirs collapses to a different shape that breaks
            submission-checker layout invariants). Same for empty orgname
            (Pitfall 1 defense-in-depth: orgname must be resolved upstream).
        ValueError: If a per-benchmark-type required field is missing or if
            any path component fails _check_safe_path_component() validation:
            - training/checkpointing: args.model.
            - vector_database: args.vdb_engine, args.vdb_index (or .index_type).
            - kv_cache: args.model.
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    args = benchmark.args

    # Defense-in-depth empty-string guards (T-1-02 + Pitfall 1).
    # Use getattr per Pitfall 2: args may not have the attribute if
    # _apply_yaml_config_overrides() dropped it via key-not-in-dict skip.
    orgname = getattr(args, 'orgname', '')
    systemname = getattr(args, 'systemname', '')
    if not orgname:
        raise ConfigurationError(
            "Cannot generate output location: orgname is empty "
            "(sentinel not resolved).",
            suggestion=(
                "Internal error: the upstream orgname-resolution gate in "
                "main._main_impl() must populate args.orgname before "
                "benchmark instantiation. If you reached this from a non-init "
                "command, run `mlpstorage init <orgname> <results-dir>` first."
            ),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )
    if not systemname:
        raise ConfigurationError(
            "Cannot generate output location: --systemname is empty.",
            suggestion=(
                "Pass --systemname <name> on the CLI or set the "
                "MLPERF_SYSTEMNAME environment variable before re-running."
            ),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )

    # Validate every user-supplied path segment that goes into the base
    # prefix and per-type tail. Blocks path-traversal ('../') and absolute-
    # path resets ('/') at the trust boundary (args/env-var → os.path.join),
    # even for callers that bypass the CLI's argparse choices= validation.
    _check_safe_path_component("mode", args.mode)
    _check_safe_path_component("orgname", orgname)
    _check_safe_path_component("systemname", systemname)
    _check_safe_path_component("datetime_str", datetime_str)

    # Shared Rules.md §2.1 prefix for every benchmark type.
    base = os.path.join(
        args.results_dir,
        args.mode,                 # closed | open | whatif
        orgname,
        "results",
        systemname,
    )

    # WR-07: all missing-required failures raise ``ConfigurationError`` (a
    # ``MLPStorageException`` subclass) so the top-level ``main()`` handler
    # surfaces them uniformly with the user-facing message + suggestion
    # (rather than reporting "Unexpected error" with a stack trace gated on
    # ``MLPS_DEBUG``).
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(args, "model"):
            raise ConfigurationError(
                "Model name is required for training benchmark output location.",
                suggestion="Pass ``--model`` (or ``-m``) on the CLI.",
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        _check_safe_path_component("model", args.model)
        _check_safe_path_component("command", args.command)
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            args.model,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        engine = getattr(args, "vdb_engine", None)
        if not engine:
            raise ConfigurationError(
                "VectorDB engine is required for output location.",
                suggestion="Pass ``--vdb-engine`` on the CLI.",
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        # Results split by index_type because AISAQ is not comparable to
        # DISKANN/HNSW — they must live in separate on-disk trees so
        # submission validation and downstream tooling never collate them
        # (per Rules.md §2.1.27).
        vdb_index = (
            getattr(args, "vdb_index", None)
            or getattr(args, "index_type", None)
        )
        if not vdb_index:
            raise ConfigurationError(
                "VectorDB index is required for output location.",
                suggestion="Pass ``--vdb-index`` on the CLI.",
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        _check_safe_path_component("vdb_engine", engine)
        _check_safe_path_component("vdb_index", vdb_index)
        _check_safe_path_component("command", args.command)
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            engine,
            vdb_index,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.kv_cache:
        model = getattr(args, "model", None)
        if not model:
            raise ConfigurationError(
                "Model is required for kv_cache output location.",
                suggestion=(
                    "Set args.model before calling generate_output_location "
                    "(KVCacheBenchmark.__init__ defaults this from "
                    "KVCACHE_MODEL_DEFAULT)."
                ),
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        _check_safe_path_component("model", model)
        _check_safe_path_component("command", args.command)
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            model,
            args.command,
            datetime_str,
        )

    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(args, "model"):
            raise ConfigurationError(
                "Model name is required for checkpointing benchmark output location.",
                suggestion="Pass ``--model`` (or ``-m``) on the CLI.",
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        _check_safe_path_component("model", args.model)
        # Checkpointing intentionally omits the <command> segment; preserves
        # the pre-refactor layout shape that downstream submission-checkers
        # already validate against.
        return os.path.join(
            base,
            benchmark.BENCHMARK_TYPE.name,
            args.model,
            datetime_str,
        )

    # WR-07: unknown BENCHMARK_TYPE used to ``print`` + ``sys.exit(1)``,
    # bypassing the logger and leaving no trail in log files. Raise a typed
    # error instead — the ``main()`` handler will log it through the normal
    # ConfigurationError flow.
    raise ConfigurationError(
        f"Unsupported benchmark type {benchmark.BENCHMARK_TYPE!r} for "
        "generate_output_location().",
        suggestion=(
            "Add the new benchmark to the if/elif chain in "
            "rules/utils.generate_output_location."
        ),
        code=ErrorCode.CONFIG_INVALID_VALUE,
    )


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
