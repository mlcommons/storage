"""
Training benchmark CLI argument builder.

This module defines the CLI arguments for the training benchmark,
including datasize, datagen, run, and configview commands.
"""

from mlpstorage_py.config import (
    MODELS, MODELS_CLOSED, MODELS_OPEN, ACCELERATORS, ACCELERATORS_CLOSED,
    DEFAULT_HOSTS, EXEC_TYPE, EXIT_CODE
)

from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_storage_type_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
    add_timeseries_arguments,
)


def add_training_arguments(parser, mode):
    """Add training benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the training benchmark.
        mode: Submission mode — one of 'closed', 'open', or 'whatif'.
    """
    model_choices = {
        "closed": MODELS_CLOSED,
        "open":   MODELS_OPEN,
        "whatif": MODELS,
    }[mode]
    accel_choices = ACCELERATORS if mode == "whatif" else ACCELERATORS_CLOSED

    # Model positional registered BEFORE subparsers — consumed before the command token
    parser.add_argument(
        "model",
        choices=model_choices,
        metavar="MODEL",
        help=HELP_MESSAGES['model']
    )

    # Subparsers AFTER the positional
    training_subparsers = parser.add_subparsers(dest="command", required=True)
    parser.required = True

    # Create subcommand parsers
    datasize = training_subparsers.add_parser(
        "datasize",
        help=HELP_MESSAGES['datasize']
    )
    datagen = training_subparsers.add_parser(
        "datagen",
        help=HELP_MESSAGES['training_datagen']
    )
    run_benchmark = training_subparsers.add_parser(
        "run",
        help=HELP_MESSAGES['run_benchmark']
    )
    configview = training_subparsers.add_parser(
        "configview",
        help=HELP_MESSAGES['configview']
    )

    for cmd_name, cmd_parser in [("datasize", datasize), ("datagen", datagen),
                                  ("run", run_benchmark), ("configview", configview)]:
        _add_training_core_args(cmd_parser, cmd_name, accel_choices)
        if mode in ("open", "whatif"):
            _add_training_open_args(cmd_parser, cmd_name)
        if mode == "whatif":
            _add_training_whatif_args(cmd_parser, cmd_name)


def _add_training_core_args(parser, command, accel_choices):
    """Add core (closed/open/whatif) training arguments to a subcommand parser.

    Args:
        parser: The subcommand parser to add arguments to.
        command: The subcommand name ('datasize', 'datagen', 'run', 'configview').
        accel_choices: Allowed accelerator type values for this mode.
    """
    # Set defaults for open-gated attrs so they always exist in the namespace
    parser.set_defaults(loops=1, params='', allow_invalid_params=False)

    add_host_arguments(parser)

    # Memory argument — not for datagen
    if command != "datagen":
        parser.add_argument(
            '--client-host-memory-in-gb', '-cm',
            type=float,
            required=True,
            help=HELP_MESSAGES['client_host_mem_GB']
        )

    # Process / accelerator count — name differs per command
    if command == "datagen":
        parser.add_argument(
            '--num-processes', '-np',
            type=int,
            required=True,
            help=HELP_MESSAGES['num_accelerators_datagen']
        )
    elif command == "datasize":
        parser.add_argument(
            '--max-accelerators', '-ma',
            type=int,
            required=True,
            help=HELP_MESSAGES['num_accelerators_datasize']
        )
    else:
        # run and configview
        parser.add_argument(
            '--num-accelerators', '-na',
            type=int,
            required=True,
            help=HELP_MESSAGES['num_accelerators_run']
        )

    # Accelerator type and num-client-hosts — for datasize and run/configview but not datagen
    if command != "datagen":
        parser.add_argument(
            '--accelerator-type', '-at',
            choices=accel_choices,
            required=True,
            help=HELP_MESSAGES['accelerator_type']
        )
        parser.add_argument(
            '--num-client-hosts', '-nc',
            type=int,
            help=HELP_MESSAGES['num_client_hosts']
        )

    parser.add_argument(
        '--exec-type', '-et',
        type=EXEC_TYPE,
        choices=list(EXEC_TYPE),
        default=EXEC_TYPE.MPI,
        help=HELP_MESSAGES['exec_type']
    )

    add_mpi_arguments(parser)

    parser.add_argument(
        '--data-dir', '-dd',
        type=str,
        help="Filesystem location for data"
    )

    add_dlio_arguments(parser)
    add_universal_arguments(parser, req_results=(command in ("run", "configview")))

    # Storage type positional for datagen, run, configview — NOT datasize
    if command in ("datagen", "run", "configview"):
        add_storage_type_arguments(parser, required=True)


def _add_training_open_args(parser, command):
    """Add open/whatif-only training arguments.

    Args:
        parser: The subcommand parser to add arguments to.
        command: The subcommand name.
    """
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


def _add_training_whatif_args(parser, command):
    """Add whatif-only training arguments.

    Args:
        parser: The subcommand parser to add arguments to.
        command: The subcommand name.
    """
    pass  # No whatif-only training args at this time


def validate_training_arguments(args):
    """Validate the whole set of args given that we're doing a training benchmark

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
