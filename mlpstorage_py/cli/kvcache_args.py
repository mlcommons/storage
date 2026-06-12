"""
KV Cache benchmark CLI argument builder.

This module defines the CLI arguments for the KV Cache benchmark,
including run and datasize commands for LLM inference storage testing.
"""

import sys

from mlpstorage_py.config import (
    KVCACHE_MODELS,
    KVCACHE_MODEL_DEFAULT,
    KVCACHE_PERFORMANCE_PROFILES,
    KVCACHE_GENERATION_MODES,
    KVCACHE_DEFAULT_DURATION,
    EXEC_TYPE,
    EXIT_CODE,
)
from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_host_arguments,
    add_mpi_arguments,
    add_timeseries_arguments,
)


# KV Cache specific help messages
KVCACHE_HELP_MESSAGES = {
    'kvcache_model': (
        "KV Cache model configuration to simulate. Determines the cache size "
        "per token and typical sequence length."
    ),
    'num_users': "Number of concurrent users to simulate for multi-tenant inference.",
    'duration': "Duration of the benchmark run in seconds.",
    'gpu_mem_gb': "GPU memory available for the first cache tier (GiB).",
    'cpu_mem_gb': "CPU memory available for the second cache tier (GiB).",
    'cache_dir': (
        "Directory path for NVMe cache tier storage. If not specified, "
        "a subdirectory in the results folder will be used."
    ),
    'generation_mode': (
        "Token generation simulation mode. Options: "
        "'none' (no generation), 'fast' (fixed rate), 'realistic' (variable rate)."
    ),
    'performance_profile': (
        "Performance profile for pass/fail criteria. "
        "'latency' optimizes for response time, 'throughput' for requests/second."
    ),
    'kvcache_run': (
        "Run the MLPerf KV Cache benchmark sequence (options 1, 2, and 3 via mlperf_wrapper.py)."
    ),
    'kvcache_datasize': (
        "Calculate memory requirements for KV cache based on model and user count."
    ),
    'disable_multi_turn': "Disable multi-turn conversation simulation.",
    'disable_prefix_caching': "Disable prefix caching optimization.",
    'enable_rag': "Enable RAG (Retrieval Augmented Generation) document handling.",
    'rag_num_docs': "Number of RAG documents per query when RAG is enabled.",
    'enable_autoscaling': "Enable autoscaling simulation for user load.",
    'autoscaler_mode': (
        "Autoscaler mode: 'qos' (quality of service based) or "
        "'predictive' (load prediction based)."
    ),
    'seed': (
        "Base random seed (default 42). Effective seed per rank = base + rank. "
        "OPEN submissions only — fixed at 42 in CLOSED."
    ),
    'kvcache_bin_path': "Path to kv-cache.py script. Auto-detected if not specified.",
    'npernode': "Number of kv-cache instances per client host (ranks per node).",
    'trials': (
        "Number of trial runs per option (default 3). "
        "OPEN submissions only — fixed at 3 in CLOSED."
    ),
    'inter_option_delay': (
        "Seconds to wait between options (default 20). "
        "OPEN submissions only — fixed at 20 in CLOSED."
    ),
    'config': (
        "Path to kv-cache config.yaml passed through to mlperf_wrapper.py. "
        "Auto-detected from wrapper's directory if not specified. "
        "OPEN submissions only — not valid in CLOSED."
    ),
}


def add_kvcache_arguments(parser, mode):
    """Add KV Cache benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the KV Cache benchmark.
        mode: One of 'closed', 'open', or 'whatif'.
    """
    kvcache_subparsers = parser.add_subparsers(dest="command", required=True)
    parser.required = True

    # Create subcommand parsers
    run_benchmark = kvcache_subparsers.add_parser(
        "run",
        help=KVCACHE_HELP_MESSAGES['kvcache_run']
    )
    datasize = kvcache_subparsers.add_parser(
        "datasize",
        help=KVCACHE_HELP_MESSAGES['kvcache_datasize']
    )

    # Add cache and universal arguments to both run and datasize commands.
    # Architectural constraint: kvcache has no file/object storage type positional.
    # Storage type args are intentionally absent from this builder.
    for _parser in [run_benchmark, datasize]:
        _add_kvcache_cache_arguments(_parser, mode)
        add_universal_arguments(_parser, req_results=(_parser is run_benchmark))

    # Run-specific arguments
    _add_kvcache_run_arguments(run_benchmark)

    if mode in ("open", "whatif"):
        _add_kvcache_model_arguments(run_benchmark)
        _add_kvcache_open_args(run_benchmark)
        _add_kvcache_mlperf_arguments(run_benchmark)

    _add_kvcache_optional_features(run_benchmark, mode)

    # Add distributed execution arguments to run command only
    _add_kvcache_distributed_arguments(run_benchmark)


def _add_kvcache_model_arguments(parser):
    """Add model configuration arguments (open/whatif only).

    In closed mode, there is no model positional — the model is fixed.
    This function is only called for open and whatif branches.

    Args:
        parser: Argparse parser to add arguments to.
    """
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        '--model', '-m',
        choices=KVCACHE_MODELS,
        default=KVCACHE_MODELS[0],
        help=KVCACHE_HELP_MESSAGES['kvcache_model']
    )
    model_group.add_argument(
        '--num-users', '-nu',
        type=int,
        default=100,
        help=KVCACHE_HELP_MESSAGES['num_users']
    )


def _add_kvcache_cache_arguments(parser, mode):
    """Add cache tier configuration arguments.

    In closed mode, gpu_mem_gb and cpu_mem_gb are fixed via set_defaults.
    In open/whatif mode, they are exposed as arguments.

    Args:
        parser: Argparse parser to add arguments to.
        mode: One of 'closed', 'open', or 'whatif'.
    """
    cache_group = parser.add_argument_group("Cache Configuration")
    cache_group.add_argument(
        '--cache-dir',
        type=str,
        help=KVCACHE_HELP_MESSAGES['cache_dir']
    )
    if mode == "closed":
        # Set defaults for all open-gated attrs so namespace attrs always exist
        cache_group.set_defaults(
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            loops=1,
            duration=KVCACHE_DEFAULT_DURATION,
            generation_mode='realistic',
            performance_profile='throughput',
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=True,
            rag_num_docs=10,
            enable_autoscaling=True,
            autoscaler_mode='qos',
            seed=42,
            trials=3,
            inter_option_delay=20,
            allow_invalid_params=False,
            params='',
        )
    else:
        cache_group.add_argument(
            '--gpu-mem-gb',
            type=float,
            default=16.0,
            help=KVCACHE_HELP_MESSAGES['gpu_mem_gb']
        )
        cache_group.add_argument(
            '--cpu-mem-gb',
            type=float,
            default=32.0,
            help=KVCACHE_HELP_MESSAGES['cpu_mem_gb']
        )


def _add_kvcache_run_arguments(parser):
    """Add core run-specific arguments (shared across all modes).

    Args:
        parser: Argparse parser to add arguments to.
    """
    run_group = parser.add_argument_group("Run Configuration")
    run_group.add_argument(
        '--kvcache-bin-path',
        type=str,
        help=KVCACHE_HELP_MESSAGES['kvcache_bin_path']
    )


def _add_kvcache_open_args(parser):
    """Add open/whatif-only KVCache run configuration arguments.

    These arguments are only available in open and whatif submission modes.
    In closed mode, their values are fixed via set_defaults in
    _add_kvcache_cache_arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    run_group = parser.add_argument_group("Open Run Configuration")
    run_group.add_argument(
        '--duration', '-d',
        type=int,
        default=KVCACHE_DEFAULT_DURATION,
        help=KVCACHE_HELP_MESSAGES['duration']
    )
    run_group.add_argument(
        '--generation-mode',
        choices=KVCACHE_GENERATION_MODES,
        default='realistic',
        help=KVCACHE_HELP_MESSAGES['generation_mode']
    )
    run_group.add_argument(
        '--performance-profile',
        choices=KVCACHE_PERFORMANCE_PROFILES,
        default='throughput',
        help=KVCACHE_HELP_MESSAGES['performance_profile']
    )
    run_group.add_argument(
        '--loops',
        type=int,
        default=1,
        help="Number of times to repeat the benchmark run"
    )
    run_group.add_argument(
        '--allow-invalid-params', '-aip',
        action='store_true',
        help="Allow parameters that would otherwise be flagged as invalid"
    )


def _add_kvcache_mlperf_arguments(parser):
    """Add MLPerf sequence arguments for the KV Cache run command.

    These arguments control the three-option MLPerf v3.0 benchmark sequence.
    In CLOSED submissions, seed/trials/inter-option-delay are fixed to their
    mandated defaults and --config is disallowed; the benchmark will hard-fail
    if the user attempts to override them.

    Args:
        parser: Argparse parser to add arguments to.
    """
    mlperf_group = parser.add_argument_group("MLPerf Sequence Configuration")
    mlperf_group.add_argument(
        '--npernode', '--num-processes-per-client',
        dest='npernode',
        type=int,
        default=1,
        help=KVCACHE_HELP_MESSAGES['npernode']
    )
    mlperf_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help=KVCACHE_HELP_MESSAGES['seed']
    )
    mlperf_group.add_argument(
        '--trials',
        type=int,
        default=None,
        help=KVCACHE_HELP_MESSAGES['trials']
    )
    mlperf_group.add_argument(
        '--inter-option-delay',
        type=int,
        default=None,
        help=KVCACHE_HELP_MESSAGES['inter_option_delay']
    )
    mlperf_group.add_argument(
        '--config',
        type=str,
        default=None,
        help=KVCACHE_HELP_MESSAGES['config']
    )


def _add_kvcache_optional_features(parser, mode):
    """Add optional feature flags.

    In closed mode, values are fixed via set_defaults.
    In open/whatif mode, flags are registered as actual arguments.

    Args:
        parser: Argparse parser to add arguments to.
        mode: One of 'closed', 'open', or 'whatif'.
    """
    features_group = parser.add_argument_group("Optional Features")
    if mode == "closed":
        features_group.set_defaults(
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=True,
            rag_num_docs=10,
            enable_autoscaling=True,
            autoscaler_mode='qos'
        )
    else:
        features_group.add_argument(
            '--disable-multi-turn',
            action='store_true',
            help=KVCACHE_HELP_MESSAGES['disable_multi_turn']
        )
        features_group.add_argument(
            '--disable-prefix-caching',
            action='store_true',
            help=KVCACHE_HELP_MESSAGES['disable_prefix_caching']
        )
        features_group.add_argument(
            '--enable-rag',
            action='store_true',
            help=KVCACHE_HELP_MESSAGES['enable_rag']
        )
        features_group.add_argument(
            '--rag-num-docs',
            type=int,
            default=10,
            help=KVCACHE_HELP_MESSAGES['rag_num_docs']
        )
        features_group.add_argument(
            '--enable-autoscaling',
            action='store_true',
            help=KVCACHE_HELP_MESSAGES['enable_autoscaling']
        )
        features_group.add_argument(
            '--autoscaler-mode',
            choices=['qos', 'predictive'],
            default='qos',
            help=KVCACHE_HELP_MESSAGES['autoscaler_mode']
        )


def _add_kvcache_distributed_arguments(parser):
    """Add distributed execution arguments for multi-host benchmarking.

    Args:
        parser: Argparse parser to add arguments to.
    """
    distributed_group = parser.add_argument_group("Distributed Execution")
    distributed_group.add_argument(
        '--exec-type', '-et',
        type=EXEC_TYPE,
        choices=list(EXEC_TYPE),
        default=EXEC_TYPE.MPI,
        help=HELP_MESSAGES['exec_type']
    )
    distributed_group.add_argument(
        '--num-processes', '-np',
        type=int,
        help="Number of MPI processes (ranks) to spawn for distributed execution."
    )

    # Add host arguments from common_args
    add_host_arguments(parser)

    # Add MPI arguments from common_args
    add_mpi_arguments(parser)

    # Add time-series arguments (open/whatif modes only)
    add_timeseries_arguments(parser)


def validate_kvcache_arguments(args):
    """Validate the whole set of args given that we're doing a kvcache benchmark

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
    error_messages = []

    if hasattr(args, 'data_access_protocol') and args.data_access_protocol != 'file':
        error_messages.append("KVCache only supports POSIX file storage, ie: --object= is not supported")

    if error_messages:
        for msg in error_messages:
            print(msg)

        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
