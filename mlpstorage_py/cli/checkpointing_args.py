"""
Checkpointing benchmark CLI argument builder.

This module defines the CLI arguments for the checkpointing benchmark,
including datasize, run, and configview commands.
"""

import sys

from mlpstorage_py.config import DEFAULT_HOSTS, EXEC_TYPE, LLM_MODELS, LLM_MODELS_CLOSED, EXIT_CODE
from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_storage_type_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
    add_timeseries_arguments,
)


def add_checkpointing_arguments(parser, mode):
    """Add checkpointing benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the checkpointing benchmark.
        mode: Submission mode — one of 'closed', 'open', or 'whatif'.
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
    configview = checkpointing_subparsers.add_parser(
        "configview",
        help=HELP_MESSAGES.get('configview', 'View final benchmark configuration')
    )

    for cmd_name, cmd_parser in [("datasize", datasize), ("run", run_benchmark),
                                  ("configview", configview)]:
        _add_checkpointing_core_args(cmd_parser, cmd_name)
        if mode in ("open", "whatif"):
            _add_checkpointing_open_args(cmd_parser, cmd_name)


def _add_checkpointing_core_args(parser, command):
    """Add core (closed/open/whatif) checkpointing arguments to a subcommand parser.

    Args:
        parser: The subcommand parser to add arguments to.
        command: The subcommand name ('datasize', 'run', 'configview').
    """
    # Set defaults for open-gated attrs so they always exist in the namespace
    parser.set_defaults(
        loops=1,
        params='',
        allow_invalid_params=False,
        num_checkpoints_read=10,
        num_checkpoints_write=10,
        dlio_bin_path=None,
        checkpoint_folder=None,
    )

    add_host_arguments(parser)

    parser.add_argument(
        '--client-host-memory-in-gb', '-cm',
        type=float,
        required=True,
        help=HELP_MESSAGES['client_host_mem_GB']
    )

    # Model as --model flag (not positional) — checkpointing keeps flag style
    parser.add_argument(
        '--model', '-m',
        choices=LLM_MODELS,
        required=True,
        help=HELP_MESSAGES['llm_model']
    )

    parser.add_argument(
        '--num-processes', '-np',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_checkpoint_accelerators']
    )

    parser.add_argument(
        '--exec-type', '-et',
        type=EXEC_TYPE,
        choices=list(EXEC_TYPE),
        default=EXEC_TYPE.MPI,
        help=HELP_MESSAGES['exec_type']
    )

    add_mpi_arguments(parser)

    # --dlio-bin-path is a deployment knob (path to the DLIO binary), not a
    # submission tunable. Rules.md does not gate it. Training already exposes it
    # in core args; checkpointing was inconsistent. Move it here so closed-mode
    # submitters can point at a custom DLIO build without flipping to open.
    add_dlio_arguments(parser)

    add_universal_arguments(parser, req_results=(command in ("run", "configview")))

    # Storage type positional for run and configview — NOT datasize
    if command in ("run", "configview"):
        add_storage_type_arguments(parser, required=True)

    # Checkpoint folder required for run only
    if command == "run":
        parser.add_argument(
            '--checkpoint-folder', '-cf',
            type=str,
            required=True,
            help=HELP_MESSAGES['checkpoint_folder']
        )


def _add_checkpointing_open_args(parser, command):
    """Add open/whatif-only checkpointing arguments.

    Args:
        parser: The subcommand parser to add arguments to.
        command: The subcommand name.
    """
    parser.add_argument(
        '--num-checkpoints-read', '-ncr',
        type=int,
        default=10,
        help=HELP_MESSAGES['num_checkpoints']
    )
    parser.add_argument(
        '--num-checkpoints-write', '-ncw',
        type=int,
        default=10,
        help=HELP_MESSAGES['num_checkpoints']
    )
    parser.add_argument(
        '--loops',
        type=int,
        default=1,
        help="Number of times to run the benchmark"
    )
    parser.add_argument(
        '--allow-invalid-params', '-aip',
        action='store_true',
        help="Allow invalid DLIO parameters to be passed"
    )
    parser.add_argument(
        '--params', '-p',
        nargs="+",
        action="append",
        default=None,  # Override set_defaults(params='') — append action requires list/None
        metavar="KEY=VALUE",
        help=HELP_MESSAGES['params']
    )
    if command == "run":
        add_timeseries_arguments(parser)


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
