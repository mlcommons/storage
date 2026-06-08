"""
VectorDB benchmark CLI argument builder.

This module defines the CLI arguments for the VectorDB benchmark,
including datasize, datagen, and run commands.
"""

from mlpstorage_py.config import (
    VECTOR_DTYPES, DISTRIBUTIONS, VECTORDB_DEFAULT_RUNTIME,
    VDB_INDEX_TYPES, VDB_INDEX_TYPES_CLOSED, VDB_BENCHMARK_MODES, EXIT_CODE
)
from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_storage_type_arguments,
    add_timeseries_arguments,
)


def add_vectordb_arguments(parser, mode):
    """Add VectorDB benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the VectorDB benchmark.
        mode: One of 'closed', 'open', or 'whatif'.
    """
    index_choices = VDB_INDEX_TYPES_CLOSED if mode == "closed" else VDB_INDEX_TYPES

    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands"
    )
    parser.required = True

    # ---- Subcommand parsers ----
    datasize = vectordb_subparsers.add_parser(
        'datasize',
        help="Calculate storage requirements for a VDB dataset"
    )
    datagen = vectordb_subparsers.add_parser(
        'datagen',
        help=HELP_MESSAGES['vdb_datagen']
    )
    run_benchmark = vectordb_subparsers.add_parser(
        'run',
        help=HELP_MESSAGES['vdb_run']
    )

    for cmd_name, cmd_parser in [("datasize", datasize), ("datagen", datagen), ("run", run_benchmark)]:
        _add_vectordb_core_args(cmd_parser, cmd_name, index_choices)
        if mode in ("open", "whatif"):
            _add_vectordb_open_args(cmd_parser, cmd_name)


def _add_vectordb_core_args(parser, command, index_choices):
    """Add core VectorDB arguments shared across all modes.

    Args:
        parser: Argparse parser to add arguments to.
        command: The subcommand name ('datasize', 'datagen', or 'run').
        index_choices: Allowed index type choices based on mode.
    """
    # Set defaults for open-gated attrs so they always exist in namespace
    parser.set_defaults(loops=1, params='', allow_invalid_params=False)

    # ---- Common arguments for datagen and run ----
    if command in ("datagen", "run"):
        parser.add_argument(
            '--host', '-s',
            type=str,
            default="127.0.0.1",
            help=HELP_MESSAGES['db_ip_address']
        )
        parser.add_argument(
            '--port', '-p',
            type=int,
            default=19530,
            help=HELP_MESSAGES['db_port']
        )
        parser.add_argument(
            '--config'
        )
        parser.add_argument(
            '--collection',
            type=str,
            help=HELP_MESSAGES['db_collection']
        )

    # ---- Datasize arguments ----
    if command == "datasize":
        parser.add_argument(
            '--dimension',
            type=int,
            default=1536,
            help=HELP_MESSAGES['dimension']
        )
        parser.add_argument(
            '--num-vectors',
            type=int,
            default=1_000_000,
            help=HELP_MESSAGES['num_vectors']
        )
        parser.add_argument(
            '--index-type',
            choices=index_choices,
            default="DISKANN",
            help="Index type for storage estimation"
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_shards']
        )
        parser.add_argument(
            '--vector-dtype',
            choices=VECTOR_DTYPES,
            default="FLOAT_VECTOR",
            help=HELP_MESSAGES['vector_dtype']
        )

    # ---- Datagen specific arguments ----
    if command == "datagen":
        parser.add_argument(
            '--dimension',
            type=int,
            default=1536,
            help=HELP_MESSAGES['dimension']
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_shards']
        )
        parser.add_argument(
            '--vector-dtype',
            choices=VECTOR_DTYPES,
            default="FLOAT_VECTOR",
            help=HELP_MESSAGES['vector_dtype']
        )
        parser.add_argument(
            '--num-vectors',
            type=int,
            default=1_000_000,
            help=HELP_MESSAGES['num_vectors']
        )
        parser.add_argument(
            '--distribution',
            choices=DISTRIBUTIONS,
            default="uniform",
            help=HELP_MESSAGES['distribution']
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1_000,
            help=HELP_MESSAGES['vdb_datagen_batch_size']
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=10_000,
            help=HELP_MESSAGES['vdb_datagen_chunk_size']
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force recreate collection if it exists"
        )

    # ---- Run specific arguments ----
    if command == "run":
        parser.add_argument(
            '--num-query-processes',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_query_processes']
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help=HELP_MESSAGES['query_batch_size']
        )
        parser.add_argument(
            '--report-count',
            type=int,
            default=100,
            help=HELP_MESSAGES['vdb_report_count']
        )
        parser.add_argument(
            '--benchmark-mode',
            dest='benchmark_mode',
            choices=VDB_BENCHMARK_MODES,
            default='timed',
            help="Benchmark mode: timed (simple_bench), query_count, or sweep (enhanced_bench)"
        )

        # End condition group for run
        end_group = parser.add_argument_group(
            "Provide an end condition of runtime (in seconds) or total number of "
            "queries to execute. The default is to run for 60 seconds"
        )
        end_condition = end_group.add_mutually_exclusive_group()
        end_condition.add_argument(
            "--runtime",
            type=int,
            help="Run for a specific duration in seconds"
        )
        end_condition.add_argument(
            "--queries",
            type=int,
            help="Run for a specific number of queries"
        )

    add_universal_arguments(parser, req_results=(command in ("datagen", "run")))

    if command in ("datagen", "run"):
        add_storage_type_arguments(parser, required=True)


def _add_vectordb_open_args(parser, command):
    """Add open/whatif-only VectorDB arguments.

    Args:
        parser: Argparse parser to add arguments to.
        command: The subcommand name.
    """
    parser.add_argument(
        '--loops',
        type=int,
        default=1,
        help="Number of times to repeat the benchmark run"
    )
    parser.add_argument(
        '--allow-invalid-params', '-aip',
        action='store_true',
        help="Allow parameters that would otherwise be flagged as invalid"
    )
    parser.add_argument(
        '--params',
        nargs="+",
        action="append",
        default=None,  # Override set_defaults(params='') — append action requires list/None
        metavar="KEY=VALUE",
        help=HELP_MESSAGES['params']
    )
    if command == "run":
        add_timeseries_arguments(parser)


def validate_vectordb_arguments(args):
    """Validate the whole set of args given that we're doing a vectordb benchmark

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
