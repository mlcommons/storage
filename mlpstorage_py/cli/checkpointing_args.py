"""
Checkpointing benchmark CLI argument builder.

This module defines the CLI arguments for the checkpointing benchmark,
including datasize and run commands.
"""

import sys

from mlpstorage_py.config import DEFAULT_HOSTS, EXEC_TYPE, LLM_MODELS, LLM_MODELS_CLOSED, EXIT_CODE
from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
    add_timeseries_arguments,
)


def add_checkpointing_arguments(parser, is_closed):
    """Add checkpointing benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the checkpointing benchmark.
    """
    checkpointing_subparsers = parser.add_subparsers(dest="command", required=True)
    parser.required = True

    # Create subcommand parsers
    datasize = checkpointing_subparsers.add_parser(
        "datasize",
        help=HELP_MESSAGES['checkpoint_datasize']
    )
    run_benchmark = checkpointing_subparsers.add_parser(
        "run",
        help=HELP_MESSAGES['checkpoint_run']
    )

    # Common arguments for both datasize and run
    for _parser in [datasize, run_benchmark]:
        add_host_arguments(_parser, is_closed)
        _parser.add_argument(
            '--client-host-memory-in-gb', '-cm',
            type=int,
            required=True,
            help=HELP_MESSAGES['client_host_mem_GB']
        )

        # Model argument - using help text with choices instead of choices param
        # to avoid very long help output
        if is_closed:
            _parser.add_argument(
                '--model', '-m',
                choices=LLM_MODELS_CLOSED,
                required=True,
                help=HELP_MESSAGES['llm_model']
            )

        else:
            _parser.add_argument(
                '--model', '-m',
                choices=LLM_MODELS,
                required=True,
                help=HELP_MESSAGES['llm_model']
            )

        if is_closed:
            _parser.set_defaults(
                num_checkpoints_read=10,
                num_checkpoints_write=10
            )
        else:
            _parser.add_argument(
                '--num-checkpoints-read', '-ncr',
                type=int,
                default=10,
                help=HELP_MESSAGES['num_checkpoints']
            )

            _parser.add_argument(
                '--num-checkpoints-write', '-ncw',
                type=int,
                default=10,
                help=HELP_MESSAGES['num_checkpoints']
            )

            add_dlio_arguments(_parser, is_closed)

            # Add exec-type and MPI arguments to both datasize and run
            _parser.add_argument(
                '--exec-type', '-et',
                type=EXEC_TYPE,
                choices=list(EXEC_TYPE),
                default=EXEC_TYPE.MPI,
                help=HELP_MESSAGES['exec_type']
            )
            add_mpi_arguments(_parser, is_closed)

    run_benchmark.add_argument(
        '--num-processes', '-np',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_checkpoint_accelerators']
    )

    run_benchmark.add_argument(
        "--checkpoint-folder", '-cf',
        type=str,
        required=True,
        help=HELP_MESSAGES['checkpoint_folder']
    )

    add_universal_arguments(run_benchmark, True, True, True, is_closed)
    add_universal_arguments(datasize, False, False, True, is_closed)

    # Add time-series arguments to run command only
    add_timeseries_arguments(run_benchmark, is_closed)


def validate_checkpointing_arguments(args):
    """Validate the whole set of args given that we're doing a checkpointing benchmark
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
    error_messages = []

    if args.model not in LLM_MODELS:
        error_messages.append("Invalid LLM model. Supported models are: {}".format(", ".join(LLM_MODELS)))
    if args.num_checkpoints_read < 0 or args.num_checkpoints_write < 0:
        error_messages.append("Number of checkpoints read and write must be non-negative")

    if error_messages:
        for msg in error_messages:
            print(msg)

        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
