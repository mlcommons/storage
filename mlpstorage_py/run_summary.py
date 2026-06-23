"""Centralized run configuration summary for MLPerf Storage.

Provides print_run_summary(args), which formats and emits a structured
table of effective Tier 1 CLI parameters and environment variables
immediately before benchmark execution.

NOTE: .env file loading happens in _apply_object_storage_params(), which
runs after run_benchmark(). This summary shows pre-.env-load env state
— by design.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

from mlpstorage_py import VERSION
from mlpstorage_py.config import (
    CONFIGS_ROOT_DIR,
    VDB_INDEX_DEFAULT,
    VECTORDB_DEFAULT_RUNTIME,
)
from mlpstorage_py.mlps_logging import setup_logging
from mlpstorage_py.storage_config import resolve_object_storage_config

logger = setup_logging("MLPerfStorage")

# Label column width
_WIDTH = 32

# Args we never print: they affect only the operator's terminal, not the
# benchmark result. Filtered out of every per-benchmark section.
_OUTPUT_ONLY_OPTIONS = frozenset({
    'quiet',
    'debug',
    'verbose',
    'stream_log_level',
})


def _row(label: str, value) -> str:
    """Return a formatted label/value row string.

    Args:
        label: Column label (left-justified to _WIDTH chars).
        value: Value to display (converted to str).

    Returns:
        Indented "  label<pad>value" string.
    """
    return f"  {label:<{_WIDTH}}{value}"


def _fmt(value: Any) -> str:
    """Render a value for the summary table.

    None/empty-string become '[not set]'. Lists render as comma-joined.
    """
    if value is None or value == '':
        return '[not set]'
    if isinstance(value, (list, tuple)):
        if not value:
            return '[empty]'
        return ', '.join(str(v) for v in value)
    return str(value)


def _append_args(lines: List[str], args, fields: List[Tuple[str, str]]) -> None:
    """Append (label, attr) rows from args to lines, applying the denylist.

    Args:
        lines: Output buffer.
        args: argparse Namespace.
        fields: Pairs of (display_label, attr_name). attr_name in
            _OUTPUT_ONLY_OPTIONS is skipped.
    """
    for label, attr in fields:
        if attr in _OUTPUT_ONLY_OPTIONS:
            continue
        lines.append(_row(label + ":", _fmt(getattr(args, attr, None))))


def _print_workload_yaml(lines: List[str], section_label: str, path: Optional[str]) -> None:
    """Append a YAML workload-config section, including full pretty-printed contents.

    Args:
        lines: Output buffer.
        section_label: Heading text (without dashes).
        path: Path to the YAML file, or None if not resolved.
    """
    lines.append("")
    lines.append(f"--- {section_label} ---")

    if not path:
        lines.append(_row("path:", "[not set]"))
        return

    lines.append(_row("path:", path))

    if not os.path.isfile(path):
        lines.append(_row("status:", "[file not found]"))
        return

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        lines.append(_row("status:", f"[unreadable: {exc}]"))
        return

    if data is None:
        lines.append(_row("status:", "[file is empty]"))
        return

    # Pretty-print with default_flow_style=False so it's block-style YAML,
    # then indent each line by two spaces so it nests under the section.
    rendered = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    lines.append("  contents:")
    for line in rendered.splitlines():
        lines.append(f"    {line}")


def _resolve_vdb_workload_config(args) -> Optional[str]:
    """Resolve the VectorDB workload-config path from args.

    Returns the first candidate that exists on disk, else the most likely
    default path (so reviewers see where the orchestrator would have looked).
    Returns None when args.config is absent and there is no default file.
    """
    name = getattr(args, 'config', None) or 'default'
    base = os.path.join(CONFIGS_ROOT_DIR, 'vectordbbench')

    candidates: List[str] = []
    if os.path.isabs(name):
        candidates.append(name)
    else:
        candidates.append(os.path.abspath(name))
        if name.endswith(('.yaml', '.yml')):
            candidates.append(os.path.join(base, name))
        else:
            candidates.append(os.path.join(base, f"{name}.yaml"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return candidates[-1] if candidates else None


def _resolve_kvcache_workload_config(args) -> Optional[str]:
    """Resolve the KVCache workload-config path from args.

    --config wins when set. Otherwise return the project-relative default
    (kv_cache_benchmark/config.yaml), which is what mlperf_wrapper.py will
    auto-detect at runtime.
    """
    explicit = getattr(args, 'config', None)
    if explicit:
        return explicit

    # Mirror _find_kvcache_script's lookup: project root is two levels up
    # from mlpstorage_py/run_summary.py.
    project_root = Path(__file__).parent.parent
    default_path = project_root / 'kv_cache_benchmark' / 'config.yaml'
    return str(default_path)


def _vdb_effective_collection(args) -> str:
    """Return the --collection value if set, else '[derived at run time]'."""
    collection = getattr(args, 'collection', None)
    if collection:
        return collection
    return '[derived from workload YAML or args at run time]'


def _vdb_effective_index(args) -> str:
    """Return the post-resolution VDB index (vdb_index | index_type | default)."""
    vdb_index = getattr(args, 'vdb_index', None)
    index_type = getattr(args, 'index_type', None)
    return str(index_type or vdb_index or VDB_INDEX_DEFAULT)


def _vdb_effective_end_condition(args) -> str:
    """Return a human-readable description of the resolved --runtime/--queries."""
    runtime = getattr(args, 'runtime', None)
    queries = getattr(args, 'queries', None)
    if queries is not None:
        return f"queries={queries}"
    if runtime is not None:
        return f"runtime={runtime}s"
    return f"runtime={VECTORDB_DEFAULT_RUNTIME}s  [default]"


def _vdb_effective_gt_collection(args) -> str:
    """Return --gt-collection if set, else the '<collection>_flat_gt' default."""
    gt = getattr(args, 'gt_collection', None)
    if gt:
        return gt
    collection = getattr(args, 'collection', None)
    if collection:
        return f"{collection}_flat_gt  [default]"
    return "<collection>_flat_gt  [default — collection unresolved]"


def _vdb_effective_recall_k(args) -> str:
    """Return --recall-k if set, else --search-limit (the default)."""
    recall_k = getattr(args, 'recall_k', None)
    if recall_k is not None:
        return str(recall_k)
    search_limit = getattr(args, 'search_limit', None)
    if search_limit is not None:
        return f"{search_limit}  [defaulted from --search-limit]"
    return '[not set]'


def _vdb_is_distributed(args) -> bool:
    """Re-implement VectorDBBenchmark._is_distributed without instantiating."""
    return bool(
        getattr(args, 'distributed', False)
        or getattr(args, 'hosts', None)
        or int(getattr(args, 'npernode', 1) or 1) > 1
    )


def _vdb_mpi_world_size(args) -> Optional[int]:
    """len(hosts) * npernode when distributed; None otherwise."""
    if not _vdb_is_distributed(args):
        return None
    hosts = getattr(args, 'hosts', None) or ['localhost']
    npernode = int(getattr(args, 'npernode', 1) or 1)
    return len(hosts) * npernode


def _kvcache_effective(args, attr: str, closed_default) -> str:
    """Return the effective KVCache value applying the None → forced-default rule.

    Mirrors kvcache._run() lines 222–225: if user passed nothing, the wrapper
    uses the closed-default. In closed mode the parser already pinned this via
    set_defaults; in open the parser default is None, and the runtime fills in.
    """
    val = getattr(args, attr, None)
    if val is None:
        return f"{closed_default}  [default]"
    return str(val)


def _kvcache_total_ranks(args) -> Optional[int]:
    """npernode * len(hosts). Returns None if either component is unknown."""
    npernode = getattr(args, 'npernode', None)
    hosts = getattr(args, 'hosts', None)
    if npernode is None:
        return None
    host_count = len(hosts) if hosts else 1
    try:
        return int(npernode) * host_count
    except (TypeError, ValueError):
        return None


def _kvcache_wrapper_path(args) -> str:
    """Path to mlperf_wrapper.py derived from kvcache_bin_path."""
    bin_path = getattr(args, 'kvcache_bin_path', None)
    if bin_path:
        return str(Path(bin_path).parent / 'mlperf_wrapper.py')
    project_root = Path(__file__).parent.parent
    return str(project_root / 'kv_cache_benchmark' / 'mlperf_wrapper.py')


# ---------------------------------------------------------------------------
# Per-benchmark sections
# ---------------------------------------------------------------------------

# Fields listed in display order. Each entry is (label, attr_name).
# attr_name in _OUTPUT_ONLY_OPTIONS is skipped by _append_args.

_VDB_FIELDS_INDEX_AND_DATASET: List[Tuple[str, str]] = [
    ("vdb_engine",                'vdb_engine'),
    ("index_type",                'index_type'),
    ("num_vectors",               'num_vectors'),
    ("dimension",                 'dimension'),
    ("num_shards",                'num_shards'),
    ("vector_dtype",              'vector_dtype'),
    ("distribution",              'distribution'),
    ("batch_size (datagen)",      'batch_size'),
    ("chunk_size",                'chunk_size'),
    ("force",                     'force'),
    ("metric_type",               'metric_type'),
    ("max_degree",                'max_degree'),
    ("search_list_size",          'search_list_size'),
    ("M (HNSW)",                  'M'),
    ("ef_construction",           'ef_construction'),
    ("inline_pq",                 'inline_pq'),
    ("monitor_interval",          'monitor_interval'),
    ("compact",                   'compact'),
]

_VDB_FIELDS_ENDPOINT: List[Tuple[str, str]] = [
    ("host (db endpoint)",        'host'),
    ("port",                      'port'),
]

_VDB_FIELDS_RUN: List[Tuple[str, str]] = [
    ("num_query_processes",       'num_query_processes'),
    ("batch_size (run)",          'batch_size'),
    ("report_count",              'report_count'),
    ("benchmark_mode",            'benchmark_mode'),
    ("vector_dim",                'vector_dim'),
    ("search_limit",              'search_limit'),
    ("search_ef",                 'search_ef'),
    ("num_query_vectors",         'num_query_vectors'),
]

_VDB_FIELDS_DISTRIBUTED: List[Tuple[str, str]] = [
    ("distributed",               'distributed'),
    ("hosts",                     'hosts'),
    ("npernode",                  'npernode'),
    ("mpi_impl",                  'mpi_impl'),
    ("coordination",              'coordination'),
    ("rank_output_dir",           'rank_output_dir'),
    ("seed",                      'seed'),
    ("ready_timeout",             'ready_timeout'),
    ("mpi_bin",                   'mpi_bin'),
    ("oversubscribe",             'oversubscribe'),
    ("allow_run_as_root",         'allow_run_as_root'),
    ("mpi_btl",                   'mpi_btl'),
    ("mpi_params",                'mpi_params'),
]

_VDB_FIELDS_TIMESERIES: List[Tuple[str, str]] = [
    ("timeseries_interval",       'timeseries_interval'),
    ("skip_timeseries",           'skip_timeseries'),
    ("max_timeseries_samples",    'max_timeseries_samples'),
]

_VDB_FIELDS_OPEN_GENERIC: List[Tuple[str, str]] = [
    ("loops",                     'loops'),
    ("params",                    'params'),
    ("allow_invalid_params",      'allow_invalid_params'),
]

_VDB_FIELDS_UNIVERSAL: List[Tuple[str, str]] = [
    ("dry_run",                   'dry_run'),
    ("verify_lockfile",           'verify_lockfile'),
    ("skip_validation",           'skip_validation'),
    ("mlpstorage_arg_overrides_file", 'config_file'),
]


def _print_vectordb_section(args, lines: List[str]) -> None:
    """Append the --- VectorDB --- block with all whatif-tier args + derived values."""
    lines.append("")
    lines.append("--- VectorDB ---")

    lines.append(_row("vdb_index (effective):", _vdb_effective_index(args)))
    lines.append(_row("collection (effective):", _vdb_effective_collection(args)))

    _append_args(lines, args, _VDB_FIELDS_INDEX_AND_DATASET)
    _append_args(lines, args, _VDB_FIELDS_ENDPOINT)

    if getattr(args, 'command', None) == 'run':
        _append_args(lines, args, _VDB_FIELDS_RUN)
        lines.append(_row("end_condition (effective):", _vdb_effective_end_condition(args)))
        lines.append(_row("gt_collection (effective):", _vdb_effective_gt_collection(args)))
        lines.append(_row("recall_k (effective):", _vdb_effective_recall_k(args)))

    _append_args(lines, args, _VDB_FIELDS_DISTRIBUTED)
    world = _vdb_mpi_world_size(args)
    if world is not None:
        lines.append(_row("mpi_world_size (derived):", world))

    if getattr(args, 'command', None) == 'run':
        _append_args(lines, args, _VDB_FIELDS_TIMESERIES)

    _append_args(lines, args, _VDB_FIELDS_OPEN_GENERIC)
    _append_args(lines, args, _VDB_FIELDS_UNIVERSAL)


_KVCACHE_FIELDS_MODEL: List[Tuple[str, str]] = [
    ("model",                     'model'),
    ("num_users",                 'num_users'),
]

_KVCACHE_FIELDS_CACHE: List[Tuple[str, str]] = [
    ("gpu_mem_gb",                'gpu_mem_gb'),
    ("cpu_mem_gb",                'cpu_mem_gb'),
]

_KVCACHE_FIELDS_RUN: List[Tuple[str, str]] = [
    ("duration",                  'duration'),
    ("generation_mode",           'generation_mode'),
    ("performance_profile",       'performance_profile'),
    ("loops",                     'loops'),
]

_KVCACHE_FIELDS_FEATURES: List[Tuple[str, str]] = [
    ("disable_multi_turn",        'disable_multi_turn'),
    ("disable_prefix_caching",    'disable_prefix_caching'),
    ("enable_rag",                'enable_rag'),
    ("rag_num_docs",              'rag_num_docs'),
    ("enable_autoscaling",        'enable_autoscaling'),
    ("autoscaler_mode",           'autoscaler_mode'),
]

_KVCACHE_FIELDS_DISTRIBUTED: List[Tuple[str, str]] = [
    ("exec_type",                 'exec_type'),
    ("num_processes",             'num_processes'),
    ("hosts",                     'hosts'),
    ("npernode",                  'npernode'),
    ("mpi_bin",                   'mpi_bin'),
    ("oversubscribe",             'oversubscribe'),
    ("allow_run_as_root",         'allow_run_as_root'),
    ("mpi_btl",                   'mpi_btl'),
    ("mpi_params",                'mpi_params'),
]

_KVCACHE_FIELDS_TIMESERIES: List[Tuple[str, str]] = [
    ("timeseries_interval",       'timeseries_interval'),
    ("skip_timeseries",           'skip_timeseries'),
    ("max_timeseries_samples",    'max_timeseries_samples'),
]

_KVCACHE_FIELDS_MISC: List[Tuple[str, str]] = [
    ("kvcache_bin_path",          'kvcache_bin_path'),
    ("allow_invalid_params",      'allow_invalid_params'),
    ("params",                    'params'),
]

_KVCACHE_FIELDS_UNIVERSAL: List[Tuple[str, str]] = [
    ("dry_run",                   'dry_run'),
    ("verify_lockfile",           'verify_lockfile'),
    ("skip_validation",           'skip_validation'),
    ("mlpstorage_arg_overrides_file", 'config_file'),
]


def _print_kvcache_section(args, lines: List[str]) -> None:
    """Append the --- KVCache --- block with all whatif-tier args + derived values."""
    lines.append("")
    lines.append("--- KVCache ---")

    _append_args(lines, args, _KVCACHE_FIELDS_MODEL)
    _append_args(lines, args, _KVCACHE_FIELDS_CACHE)

    cache_dir = getattr(args, 'cache_dir', None)
    if cache_dir:
        lines.append(_row("cache_dir:", cache_dir))
    else:
        lines.append(_row("cache_dir (effective):", "<results_dir>/.../kvcache_cache  [default]"))

    _append_args(lines, args, _KVCACHE_FIELDS_RUN)

    # MLPerf sequence — show effective seed/trials/inter_option_delay
    # (the closed-mode mandate; open-mode default if user passed None).
    lines.append(_row("seed (effective):",               _kvcache_effective(args, 'seed', 42)))
    lines.append(_row("trials (effective):",             _kvcache_effective(args, 'trials', 3)))
    lines.append(_row("inter_option_delay (effective):", _kvcache_effective(args, 'inter_option_delay', 20)))

    total_ranks = _kvcache_total_ranks(args)
    if total_ranks is not None:
        lines.append(_row("total_ranks (derived):", total_ranks))

    _append_args(lines, args, _KVCACHE_FIELDS_FEATURES)
    _append_args(lines, args, _KVCACHE_FIELDS_DISTRIBUTED)
    _append_args(lines, args, _KVCACHE_FIELDS_TIMESERIES)
    _append_args(lines, args, _KVCACHE_FIELDS_MISC)

    lines.append(_row("wrapper_path (derived):", _kvcache_wrapper_path(args)))

    _append_args(lines, args, _KVCACHE_FIELDS_UNIVERSAL)


def print_run_summary(args) -> None:
    """Print a structured table of Tier 1 CLI args and env vars via logger.status().

    The summary is printed immediately before benchmark execution.  When the
    data_access_protocol is 'object', a second section with S3 environment
    variable values is appended. When the benchmark is 'vectordb' or 'kvcache',
    a per-benchmark argument section is appended with the full whatif-tier
    surface plus the derived/effective values that drive the run.

    Credentials are never shown as plain text — resolve_object_storage_config()
    pre-redacts them before returning.

    Args:
        args: argparse Namespace (or compatible object).  All attribute access
              uses getattr() with safe defaults so this function is safe to call
              regardless of which subcommand populated ``args``.
    """
    # Guard: suppress entirely when --quiet is passed.
    if getattr(args, 'quiet', False):
        return

    benchmark = getattr(args, 'benchmark', None)

    lines = ["", f"--- Run Configuration (mlpstorage {VERSION}) ---"]

    # Tier 1 CLI args — use getattr so absent attrs are '[not set]' not AttributeError.
    # For benchmarks that don't use accelerator-related knobs (VDB, KVCache), drop
    # those rows so reviewers don't see noise.
    _tier1_all = [
        ("benchmark",                 'benchmark'),
        ("command",                   'command'),
        ("mode",                      'mode'),
        ("data_dir",                  'data_dir'),
        ("results_dir",               'results_dir'),
        ("data_access_protocol",      'data_access_protocol'),
        ("num_accelerators",          'num_accelerators'),
        ("num_processes",             'num_processes'),
        ("accelerator_type",          'accelerator_type'),
        ("client_host_memory_in_gb",  'client_host_memory_in_gb'),
        ("hosts",                     'hosts'),
        ("exec_type",                 'exec_type'),
        ("mpi_bin",                   'mpi_bin'),
        ("loops",                     'loops'),
    ]
    _accel_only = {'num_accelerators', 'accelerator_type', 'client_host_memory_in_gb'}
    for label, attr in _tier1_all:
        if benchmark in ('vectordb', 'kvcache') and attr in _accel_only:
            continue
        lines.append(_row(label + ":", _fmt(getattr(args, attr, None))))

    # Always-visible environment section.
    lines.append("")
    lines.append("--- Environment ---")
    lines.append(_row("MLPERF_RESULTS_DIR:", os.environ.get('MLPERF_RESULTS_DIR', '[not set]')))
    lines.append(_row("MPI_RUN_BIN:",        os.environ.get('MPI_RUN_BIN',        '[not set]')))
    lines.append(_row("MPI_EXEC_BIN:",       os.environ.get('MPI_EXEC_BIN',       '[not set]')))
    # KVCACHE_SELECTED_WORKLOADS is read by kv-cache-wrapper.sh and filters the
    # workload set. Surface it for kvcache so reviewers know whether the full
    # suite or a subset ran.
    if benchmark == 'kvcache':
        lines.append(_row(
            "KVCACHE_SELECTED_WORKLOADS:",
            os.environ.get('KVCACHE_SELECTED_WORKLOADS', '[not set]'),
        ))

    # Per-benchmark argument section.
    if benchmark == 'vectordb':
        _print_vectordb_section(args, lines)
        _print_workload_yaml(
            lines,
            'VectorDB Workload Config',
            _resolve_vdb_workload_config(args),
        )
    elif benchmark == 'kvcache':
        _print_kvcache_section(args, lines)
        _print_workload_yaml(
            lines,
            'KVCache Workload Config',
            _resolve_kvcache_workload_config(args),
        )

    # Object storage section — only when protocol is explicitly 'object'.
    if getattr(args, 'data_access_protocol', None) == 'object':
        config = resolve_object_storage_config()
        endpoint_val, endpoint_src = config['endpoint']
        if endpoint_val:
            endpoint_display = f"{endpoint_val}  [from {endpoint_src}]"
        else:
            endpoint_display = '[not set]'

        lines.append("")
        lines.append("--- Object Storage (S3) ---")
        lines.append(_row("bucket:",                config['bucket'] or '[not set]'))
        lines.append(_row("storage_library:",       config['storage_library']))
        lines.append(_row("uri_scheme:",            config['uri_scheme']))
        lines.append(_row("endpoint:",              endpoint_display))
        lines.append(_row("load_balance_strategy:", config['load_balance_strategy']))
        lines.append(_row("aws_region:",            config['aws_region']))
        lines.append(_row("aws_ca_bundle:",         config['aws_ca_bundle'] or '[not set]'))
        lines.append(_row("AWS_ACCESS_KEY_ID:",     config['aws_access_key_id_redacted']))
        lines.append(_row("AWS_SECRET_ACCESS_KEY:", config['aws_secret_access_key_redacted']))

    lines.append("")

    for line in lines:
        logger.status(line)
