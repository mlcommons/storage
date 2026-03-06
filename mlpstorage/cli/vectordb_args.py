"""
VectorDB benchmark CLI argument builder.

This module defines the CLI arguments for the VectorDB benchmark,
including datagen and run commands.
"""

from mlpstorage.config import VECTOR_DTYPES, DISTRIBUTIONS, VECTORDB_DEFAULT_RUNTIME
from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_timeseries_arguments,
)


def add_vectordb_arguments(parser):
    """Add VectorDB benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the VectorDB benchmark.
    """
    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands"
    )
    parser.required = True

    # Create subcommand parsers
    datagen = vectordb_subparsers.add_parser(
        'datagen',
        help=HELP_MESSAGES['vdb_datagen']
    )
    run_benchmark = vectordb_subparsers.add_parser(
        'run',
        help=HELP_MESSAGES['vdb_run']
    )

    # Common arguments for both datagen and run
    for _parser in [datagen, run_benchmark]:
        _parser.add_argument(
            '--host', '-s',
            type=str,
            default="127.0.0.1",
            help=HELP_MESSAGES['db_ip_address']
        )
        _parser.add_argument(
            '--port', '-p',
            type=int,
            default=19530,
            help=HELP_MESSAGES['db_port']
        )
        _parser.add_argument(
            '--config'
        )
        _parser.add_argument(
            '--collection',
            type=str,
            help=HELP_MESSAGES['db_collection']
        )

    # Datagen specific arguments
    datagen.add_argument(
        '--dimension',
        type=int,
        default=1536,
        help=HELP_MESSAGES['dimension']
    )
    datagen.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help=HELP_MESSAGES['num_shards']
    )
    datagen.add_argument(
        '--vector-dtype',
        choices=VECTOR_DTYPES,
        default="FLOAT_VECTOR",
        help=HELP_MESSAGES['vector_dtype']
    )
    datagen.add_argument(
        '--num-vectors',
        type=int,
        default=1_000_000,
        help=HELP_MESSAGES['num_vectors']
    )
    datagen.add_argument(
        '--distribution',
        choices=DISTRIBUTIONS,
        default="uniform",
        help=HELP_MESSAGES['distribution']
    )
    datagen.add_argument(
        '--batch-size',
        type=int,
        default=1_000,
        help=HELP_MESSAGES['vdb_datagen_batch_size']
    )
    datagen.add_argument(
        '--chunk-size',
        type=int,
        default=10_000,
        help=HELP_MESSAGES['vdb_datagen_chunk_size']
    )
    datagen.add_argument(
        "--force",
        action="store_true",
        help="Force recreate collection if it exists"
    )

    # Run specific arguments
    run_benchmark.add_argument(
        '--num-query-processes',
        type=int,
        default=1,
        help=HELP_MESSAGES['num_query_processes']
    )
    run_benchmark.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help=HELP_MESSAGES['query_batch_size']
    )
    run_benchmark.add_argument(
        '--report-count',
        type=int,
        default=100,
        help=HELP_MESSAGES['vdb_report_count']
    )

    # End condition group for run
    end_group = run_benchmark.add_argument_group(
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

    # Add universal arguments to all subcommands
    for _parser in [datagen, run_benchmark]:
        add_universal_arguments(_parser)

    # Add time-series arguments to run command only
    add_timeseries_arguments(run_benchmark)
