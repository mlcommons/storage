"""
Training benchmark CLI argument builder.

This module defines the CLI arguments for the training benchmark,
including datasize, datagen, run, and configview commands.
"""

import argparse
import sys

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


def _positive_int(raw: str) -> int:
    """argparse type: accept positive ints; reject 0, negative, and non-numeric.

    Used by --drop-caches-timeout-seconds.  DLIO clamps to a minimum of 1
    on its end too, but rejecting at the CLI boundary gives a clearer error
    message than `subprocess.run(timeout=0)` would later.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected positive integer, got {raw!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer (>= 1), got {value}")
    return value


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

    # --data-dir is optional at the argparse layer so --config-file (applied
    # after parse_args) can supply it for object-storage workflows. Enforcement
    # is split: file mode is checked in cli_parser before YAML overrides so
    # users get an immediate, argparse-style error; object mode is checked in
    # validate_training_arguments after YAML has had its chance to populate it.
    parser.add_argument(
        '--data-dir', '-dd',
        type=str,
        help=(
            "Dataset location. For file storage, this is a filesystem path. "
            "For object storage, this is an object key prefix or full object URI."
        )
    )

    add_dlio_arguments(parser)

    # Training `run` only: per-call timeout for DLIO's per-epoch page-cache
    # flush.  Deployment knob (not a submission tunable), so it's exposed in
    # every mode like --dlio-bin-path.  Plumbed through to DLIO via the
    # DLIO_DROP_CACHES_TIMEOUT env var.  See mlcommons/storage #487.
    if command == "run":
        parser.add_argument(
            '--drop-caches-timeout-seconds',
            type=_positive_int,
            default=None,
            metavar='SECONDS',
            help=(
                "Per-call timeout for the per-epoch page-cache flush "
                "(`sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`). "
                "Default is DLIO's built-in 30s.  Raise this on large-RAM "
                "hosts where the kernel needs longer to drop caches "
                "(e.g. 300).  Plumbed through to DLIO via the "
                "DLIO_DROP_CACHES_TIMEOUT env var."
            ),
        )

    # --o-direct: available for datagen, run, and configview (not datasize).
    # Routes all training I/O through s3dlio's direct:// URI scheme, opening
    # every file with O_DIRECT so reads bypass the OS page cache entirely.
    # Works for ALL training workloads regardless of data format — this is
    # independent of reader.odirect, which is the legacy NPY/NPZ-only path.
    # Incompatible with --object (O_DIRECT targets local filesystem only).
    # See mlcommons/storage#507.
    if command != 'datasize':
        parser.add_argument(
            '--o-direct',
            action='store_true',
            default=False,
            dest='o_direct',
            help=(
                "Route all training I/O through s3dlio's O_DIRECT local "
                "filesystem mode (direct:// URI scheme), bypassing the OS "
                "page cache.  Works for every training workload regardless "
                "of data format.  Incompatible with --object (O_DIRECT "
                "targets the local filesystem only)."
            ),
        )

    # --params is allowed in CLOSED mode for the parameters listed in
    # TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS (e.g. dataset.num_files_train,
    # dataset.num_subfolders_train). Register it in the core args so closed
    # submissions can actually pass those overrides — gating to open/whatif
    # caused #433. `--param` (singular) is kept as a legacy alias so older
    # docs and the strings emitted by `datasize` still parse.
    parser.add_argument(
        '--params', '--param', '-p',
        dest='params',
        nargs="+",
        action="append",
        default=None,  # append action requires list/None; set_defaults(params='') is overridden here
        metavar="KEY=VALUE",
        help=HELP_MESSAGES['params']
    )

    add_universal_arguments(
        parser,
        req_results=(command in ("run", "configview")),
        # D-10: training datagen/run/configview/datasize are all emitting
        # commands and require --systemname (LAY-04).
        req_systemname=(command in ("datagen", "run", "configview", "datasize")),
    )

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
    command = getattr(args, 'command', None)
    protocol = getattr(args, 'data_access_protocol', None)

    # Object-mode --data-dir enforcement runs here (post-YAML) so --config-file
    # can populate data_dir for object workflows. File-mode enforcement happens
    # earlier in cli_parser.parse_arguments via parser.error().
    if command in ('datagen', 'run') and protocol == 'object' and not getattr(args, 'data_dir', None):
        print(
            f"ERROR: --data-dir is required for training {command} with object storage.\n"
            "  Specify --data-dir <key-prefix-or-URI> on the command line, or set\n"
            "  'data_dir:' in the file passed via --config-file.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)

    if getattr(args, 'o_direct', False) and protocol == 'object':
        print(
            "ERROR: --o-direct is incompatible with --object.\n"
            "  --o-direct routes I/O through s3dlio's direct:// URI scheme, which\n"
            "  reads from the local filesystem with O_DIRECT — not from an S3 endpoint.\n"
            "  Use --file with --o-direct, or use --object without --o-direct.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
