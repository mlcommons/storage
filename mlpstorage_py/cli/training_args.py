"""
Training benchmark CLI argument builder.

This module defines the CLI arguments for the training benchmark,
including datasize, datagen, run, and configview commands.
"""

from mlpstorage_py.config import (
    MODELS, MODELS_CLOSED, ACCELERATORS, ACCELERATORS_CLOSED,
    DEFAULT_HOSTS, EXEC_TYPE, EXIT_CODE
)

from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
    add_timeseries_arguments,
)


def add_training_arguments(parser, is_closed):
    """Add training benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the training benchmark.
    """
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

    # Common arguments for datasize, datagen, and run
    for _parser in [datasize, datagen, run_benchmark]:
        add_host_arguments(_parser, is_closed)
        if is_closed:
            _parser.add_argument(
                '--model', '-m',
                choices=MODELS_CLOSED,
                required=True,
                help=HELP_MESSAGES['model']
            )
        else:
            _parser.add_argument(
                '--model', '-m',
                choices=MODELS,
                required=True,
                help=HELP_MESSAGES['model']
            )

        # Memory argument (not for datagen)
        if _parser != datagen:
            _parser.add_argument(
                '--client-host-memory-in-gb', '-cm',
                type=int,
                required=True,
                help=HELP_MESSAGES['client_host_mem_GB']
            )

        _parser.add_argument(
            '--exec-type', '-et',
            type=EXEC_TYPE,
            choices=list(EXEC_TYPE),
            default=EXEC_TYPE.MPI,
            help=HELP_MESSAGES['exec_type']
        )

        add_mpi_arguments(_parser, is_closed)

    # Command-specific process count arguments
    datagen.add_argument(
        '--num-processes', '-np',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_accelerators_datagen']
    )
    datasize.add_argument(
        '--max-accelerators', '-ma',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_accelerators_datasize']
    )
    run_benchmark.add_argument(
        '--num-accelerators', '-na',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_accelerators_run']
    )
    configview.add_argument(
        '--num-accelerators', '-na',
        type=int,
        required=True,
        help=HELP_MESSAGES['num_accelerators_run']
    )

    # Accelerator type and num client hosts for datasize and run
    for _parser in [datasize, run_benchmark]:
        if is_closed:
            _parser.add_argument(
                '--accelerator-type', '-g',
                choices=ACCELERATORS_CLOSED,
                required=True,
                help=HELP_MESSAGES['accelerator_type']
            )
        else:
            _parser.add_argument(
                '--accelerator-type', '-g',
                choices=ACCELERATORS,
                required=True,
                help=HELP_MESSAGES['accelerator_type']
            )
        _parser.add_argument(
            '--num-client-hosts', '-nc',
            type=int,
            help=HELP_MESSAGES['num_client_hosts']
        )

    # Common arguments for all training subcommands
    for _parser in [datasize, datagen, run_benchmark, configview]:
        _parser.add_argument(
            "--data-dir", '-dd',
            type=str,
            help="Filesystem location for data"
        )
        add_dlio_arguments(_parser, is_closed)

    add_universal_arguments(datasize, False, False, False, is_closed)
    add_universal_arguments(datagen, True, True, True, is_closed)
    add_universal_arguments(run_benchmark, True, True, True, is_closed)
    add_universal_arguments(configview, False, True, True, is_closed)

    # Add time-series arguments to run command only
    add_timeseries_arguments(run_benchmark, is_closed)


def validate_training_arguments(args):
    """Validate the whole set of args given that we're doing a training benchmark
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
