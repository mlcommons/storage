"""
Checkpointing benchmark CLI argument builder.

This module defines the CLI arguments for the checkpointing benchmark,
including datasize and run commands.
"""

from mlpstorage.config import DEFAULT_HOSTS, EXEC_TYPE
from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
)


def add_checkpointing_arguments(parser):
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
        add_host_arguments(_parser)

        _parser.add_argument(
            '--client-host-memory-in-gb', '-cm',
            type=int,
            required=True,
            help=HELP_MESSAGES['client_host_mem_GB']
        )

        # Model argument - using help text with choices instead of choices param
        # to avoid very long help output
        _parser.add_argument(
            '--model', '-m',
            required=True,
            help=HELP_MESSAGES['llm_model']
        )

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

        _parser.add_argument(
            '--num-processes', '-np',
            type=int,
            required=True,
            help=HELP_MESSAGES['num_checkpoint_accelerators']
        )

        _parser.add_argument(
            "--checkpoint-folder", '-cf',
            type=str,
            required=True,
            help=HELP_MESSAGES['checkpoint_folder']
        )

        add_dlio_arguments(_parser)

        # MPI options only for run command
        if _parser == run_benchmark:
            _parser.add_argument(
                '--exec-type', '-et',
                type=EXEC_TYPE,
                choices=list(EXEC_TYPE),
                default=EXEC_TYPE.MPI,
                help=HELP_MESSAGES['exec_type']
            )
            add_mpi_arguments(_parser)

        add_universal_arguments(_parser)
