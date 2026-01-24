"""
KV Cache benchmark CLI argument builder.

This module defines the CLI arguments for the KV Cache benchmark,
including run and datasize commands for LLM inference storage testing.
"""

from mlpstorage.config import (
    KVCACHE_MODELS,
    KVCACHE_PERFORMANCE_PROFILES,
    KVCACHE_GENERATION_MODES,
    KVCACHE_DEFAULT_DURATION,
    EXEC_TYPE,
)
from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_host_arguments,
    add_mpi_arguments,
)


# KV Cache specific help messages
KVCACHE_HELP_MESSAGES = {
    'kvcache_model': (
        "KV Cache model configuration to simulate. Determines the cache size "
        "per token and typical sequence length."
    ),
    'num_users': "Number of concurrent users to simulate for multi-tenant inference.",
    'duration': "Duration of the benchmark run in seconds.",
    'gpu_mem_gb': "GPU memory available for the first cache tier (GB).",
    'cpu_mem_gb': "CPU memory available for the second cache tier (GB).",
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
        "Run the KV Cache benchmark simulating LLM inference storage workload."
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
    'seed': "Random seed for reproducible benchmark runs.",
    'kvcache_bin_path': "Path to kv-cache.py script. Auto-detected if not specified.",
}


def add_kvcache_arguments(parser):
    """Add KV Cache benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the KV Cache benchmark.
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

    # Add arguments to both run and datasize commands
    for _parser in [run_benchmark, datasize]:
        _add_kvcache_model_arguments(_parser)
        _add_kvcache_cache_arguments(_parser)
        add_universal_arguments(_parser)

    # Run-specific arguments
    _add_kvcache_run_arguments(run_benchmark)
    _add_kvcache_optional_features(run_benchmark)

    # Add distributed execution arguments to run command only
    _add_kvcache_distributed_arguments(run_benchmark)


def _add_kvcache_model_arguments(parser):
    """Add model configuration arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        '--model', '-m',
        choices=KVCACHE_MODELS,
        default='llama3.1-8b',
        help=KVCACHE_HELP_MESSAGES['kvcache_model']
    )
    model_group.add_argument(
        '--num-users', '-nu',
        type=int,
        default=100,
        help=KVCACHE_HELP_MESSAGES['num_users']
    )


def _add_kvcache_cache_arguments(parser):
    """Add cache tier configuration arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    cache_group = parser.add_argument_group("Cache Configuration")
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
    cache_group.add_argument(
        '--cache-dir',
        type=str,
        help=KVCACHE_HELP_MESSAGES['cache_dir']
    )


def _add_kvcache_run_arguments(parser):
    """Add run-specific arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    run_group = parser.add_argument_group("Run Configuration")
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
        default='latency',
        help=KVCACHE_HELP_MESSAGES['performance_profile']
    )
    run_group.add_argument(
        '--seed',
        type=int,
        help=KVCACHE_HELP_MESSAGES['seed']
    )
    run_group.add_argument(
        '--kvcache-bin-path',
        type=str,
        help=KVCACHE_HELP_MESSAGES['kvcache_bin_path']
    )


def _add_kvcache_optional_features(parser):
    """Add optional feature flags.

    Args:
        parser: Argparse parser to add arguments to.
    """
    features_group = parser.add_argument_group("Optional Features")
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
