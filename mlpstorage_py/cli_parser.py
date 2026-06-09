"""
CLI argument parsing for MLPerf Storage benchmarks.

This module provides the main argument parsing entry point,
using modular argument builders from the cli package.
"""

import argparse
import re
import shlex
import sys

from mlpstorage_py import VERSION
from mlpstorage_py.config import LLM_MODELS, VECTORDB_DEFAULT_RUNTIME, EXIT_CODE

# Import modular argument builders from cli package
from mlpstorage_py.cli import (
    HELP_MESSAGES,
    PROGRAM_DESCRIPTIONS,
    MLPStorageHelpFormatter,
    add_universal_arguments,
    add_training_arguments,      validate_training_arguments,
    add_checkpointing_arguments, validate_checkpointing_arguments,
    add_vectordb_arguments,      validate_vectordb_arguments,
    add_kvcache_arguments,       validate_kvcache_arguments,
    add_reports_arguments,
    add_history_arguments,
    add_lockfile_arguments,
    add_version_arguments,
)

# Backwards compatibility aliases
help_messages = HELP_MESSAGES
prog_descriptions = PROGRAM_DESCRIPTIONS


def _apply_formatter(parser):
    """Recursively set MLPStorageHelpFormatter on every parser in the subparser tree."""
    parser.formatter_class = MLPStorageHelpFormatter
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _apply_formatter(subparser)


def _build_mode_branch(mode_parser, mode):
    """Build the benchmark subparser tree for a given mode (closed/open/whatif).

    Args:
        mode_parser: The argparse subparser for this mode.
        mode: One of 'closed', 'open', 'whatif'.
    """
    benchmarks = mode_parser.add_subparsers(dest="benchmark", required=True)

    training_parser = benchmarks.add_parser(
        "training",
        help="Training benchmark (unet3d, retinanet)"
    )
    checkpointing_parser = benchmarks.add_parser(
        "checkpointing",
        help="Checkpointing benchmark (llama3-8b, llama3-70b, etc.)"
    )
    vectordb_parser = benchmarks.add_parser(
        "vectordb",
        help="Vector database benchmark (PREVIEW)"
    )
    kvcache_parser = benchmarks.add_parser(
        "kvcache",
        help="KV-cache benchmark for LLM inference"
    )

    add_training_arguments(training_parser, mode)
    add_checkpointing_arguments(checkpointing_parser, mode)
    add_vectordb_arguments(vectordb_parser, mode)
    add_kvcache_arguments(kvcache_parser, mode)


def parse_arguments():
    """Parse command-line arguments for MLPerf Storage benchmarks.

    Returns:
        argparse.Namespace: Parsed and validated arguments.
    """
    _argv = sys.argv[1:]

    # HELP-01: --help_all — print full command tree and exit
    if '--help_all' in _argv:
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
        print(HELP_ALL_TEXT)
        sys.exit(0)

    # HELP-02 / HELP-03: context-sensitive help — bare, --help, AND incomplete paths
    # R-03-01 fix: call get_context_help_tokens unconditionally (not gated on --help presence).
    # Strip help flags first so they don't appear as positionals. Then strip all remaining
    # option-style tokens (anything starting with '-') so that flags like '-cm 64' interspersed
    # between positionals don't confuse the path lookup.
    _help_flags = {'-h', '--help'}
    _stripped = [a for a in _argv if a not in _help_flags]
    _positionals = [a for a in _stripped if not a.startswith('-')]
    from mlpstorage_py.cli.help_formatter import get_context_help_tokens, SYNOPSIS_TEXT
    _msg = get_context_help_tokens(_positionals)
    if _msg is not None:
        # Fire for: bare invocation, --help at any level, AND bare incomplete paths
        # (e.g., 'mlpstorage closed training' with no --help still shows "next: unet3d | retinanet")
        if _help_flags.intersection(_argv):
            print(SYNOPSIS_TEXT)
            print()
        print(_msg + '  (or -h or --help_all for details)')
        sys.exit(0)
    # _msg is None → leaf level OR unrecognized token → fall through to argparse (HELP-03)

    parser = argparse.ArgumentParser(
        prog="mlpstorage",
        description="Script to launch the MLPerf Storage benchmark"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    # NOTE: VERSION currently returns a wrong string (mlpstorage_py dist name bug).
    # This will be fixed in Phase 2. Do not add logic here to work around it.

    top = parser.add_subparsers(dest="mode", required=True)

    # Three benchmark mode branches
    for mode_name in ("closed", "open", "whatif"):
        mode_parser = top.add_parser(
            mode_name,
            description=f"Run benchmarks in {mode_name} configuration",
            help=f"{mode_name.capitalize()} submission mode"
        )
        _build_mode_branch(mode_parser, mode_name)

    # Utility siblings — top-level, not nested under modes
    reports_parser = top.add_parser("reports", help="Generate a report from benchmark results")
    history_parser = top.add_parser("history", help="Display benchmark history")
    lockfile_parser = top.add_parser("lockfile", help="Generate and verify package lockfiles")
    version_parser = top.add_parser("version", description="Print the mlpstorage package version", help="Show installed package version and exit")
    add_reports_arguments(reports_parser)
    add_history_arguments(history_parser)
    add_lockfile_arguments(lockfile_parser)
    add_version_arguments(version_parser)

    _apply_formatter(parser)

    parsed_args = parser.parse_args()

    # NOTE: No post-parse consolidation for data_access_protocol here.
    # add_storage_type_arguments() registers 'data_access_protocol' as a positional;
    # argparse sets it directly to 'file'|'object'|None. The old --file/--object
    # consolidation block is removed entirely.

    # Apply YAML config file overrides if specified
    if hasattr(parsed_args, 'config_file') and parsed_args.config_file:
        parsed_args = apply_yaml_config_overrides(parsed_args)

    validate_args(parsed_args)
    return parsed_args


def apply_yaml_config_overrides(args):
    """
    Apply overrides from a YAML config file to the parsed arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments

    Returns:
        argparse.Namespace: The updated arguments with YAML overrides applied
    """
    import yaml

    try:
        with open(args.config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if not yaml_config:
            print(f"Warning: Config file {args.config_file} is empty or invalid")
            return args

        # Convert args to a dictionary for easier manipulation
        args_dict = vars(args)

        # Apply overrides from YAML
        for key, value in yaml_config.items():
            # Skip if the key doesn't exist in args
            if key not in args_dict:
                print(f"Warning: Config file contains unknown parameter '{key}', skipping")
                continue

            # Skip if the value is None (to avoid overriding CLI args with None)
            if value is None:
                continue

            # Handle special cases for list arguments
            if isinstance(args_dict.get(key), list) and not isinstance(value, list):
                if key == 'hosts':
                    # Convert string to list for hosts
                    args_dict[key] = value.split(',')
                elif key == 'params':
                    # Convert dict to list of "key=value" strings for params
                    if isinstance(value, dict):
                        args_dict[key] = [f"{k}={v}" for k, v in value.items()]
                    else:
                        print(f"Warning: Invalid format for 'params' in config file, skipping")
                        continue
            else:
                # Regular case - just override the value
                args_dict[key] = value

        # Convert back to Namespace
        return argparse.Namespace(**args_dict)

    except FileNotFoundError:
        print(f"Error: Config file {args.config_file} not found")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
    except Exception as e:
        print(f"Error applying config file overrides: {e}")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)

# These are used by the history tracker to know if logging needs to be updated.
logging_options = ['debug', 'verbose', 'stream_log_level']


def validate_args(args):
    """Validate the whole set of args for the different arg suites

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
    if getattr(args, 'mode', None) == 'version':
        return
    benchmark = getattr(args, 'benchmark', None)
    if benchmark == 'training':
        validate_training_arguments(args)
    if benchmark == 'checkpointing':
        validate_checkpointing_arguments(args)
    if benchmark == 'vectordb':
        validate_vectordb_arguments(args)
    if benchmark == 'kvcache':
        validate_kvcache_arguments(args)


def update_args(args):
    """
    This method is an interface between the CLI and the benchmark class.
    """
    if not hasattr(args, 'num_processes'):
        # Different commands for training use different nomenclature for the number of mpi processes to use
        # Training = num_accelerators
        # Datasize = max_accelerators
        # Datagen = num_processes
        # Checkpoint = num_processes
        # We want to consistently use num_processes in code but the different options for the CLI
        for arg in ['num_processes', 'num_accelerators', 'max_accelerators']:
            if hasattr(args, arg) and type(getattr(args, arg)) is int:
                print(f'Setting attr from {arg} to {getattr(args, arg)}')
                setattr(args, 'num_processes', int(getattr(args, arg)))
                break

    if hasattr(args, 'runtime') and hasattr(args, 'queries'):
        # For VectorDB we need runtime or queries. If none defined use a default runtime
        if not args.runtime and not args.queries:
            args.runtime = VECTORDB_DEFAULT_RUNTIME  # Default runtime if not provided

    # Check for list of lists in params and flatten them
    if hasattr(args, 'params') and args.params:
        flattened_params = [item for sublist in args.params for item in sublist]
        setattr(args, 'params', flattened_params)

    if hasattr(args, 'mpi_params') and args.mpi_params:
        # --mpi-params is collected with action="append" as a list of raw
        # strings, each potentially containing several space-separated MPI
        # flags, e.g. ["-genv PMI_VERSION=2 -genv FI_PROVIDER=tcp"].
        # MPI flags begin with '-', so nargs="+" used to reject them with
        # "expected at least one argument" (see issue #422). We now accept the
        # whole string and tokenize it here with shlex so quoting is honored
        # and downstream (generate_mpi_prefix_cmd) receives a flat token list.
        flattened_mpi_params = []
        for chunk in args.mpi_params:
            if isinstance(chunk, (list, tuple)):
                # Backwards-compat: tolerate the old nested-list shape.
                for item in chunk:
                    flattened_mpi_params.extend(shlex.split(item))
            else:
                flattened_mpi_params.extend(shlex.split(chunk))
        setattr(args, 'mpi_params', flattened_mpi_params)

    if hasattr(args, 'hosts') and args.hosts is not None:
        # Accept any of the following equivalent forms and normalize to a clean list:
        #   --hosts h1 h2 h3              -> ['h1', 'h2', 'h3']
        #   --hosts h1,h2,h3              -> ['h1', 'h2', 'h3']
        #   --hosts 'h1 h2 h3'            -> ['h1', 'h2', 'h3']   (quoted, e.g. from YAML)
        #   --hosts='h1,h2,h3'            -> ['h1', 'h2', 'h3']   (DLIO subprocess form)
        #   --hosts='h1 h2 h3'            -> ['h1', 'h2', 'h3']   (quoted after '=')
        # This defends against the argparse + nargs='+' + '=' interaction documented in
        # https://github.com/mlcommons/storage/issues/322.
        raw = args.hosts if isinstance(args.hosts, list) else [args.hosts]
        normalized = []
        for item in raw:
            if not isinstance(item, str):
                continue
            for tok in re.split(r'[,\s]+', item.strip()):
                if tok:
                    normalized.append(tok)
        if not normalized:
            print("ERROR: --hosts is empty after parsing", file=sys.stderr)
            sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
        args.hosts = normalized

    if hasattr(args, 'hosts') and getattr(args, 'num_client_hosts', None) is None:
        setattr(args, "num_client_hosts", len(args.hosts))


if __name__ == "__main__":
    args = parse_arguments()
    import pprint
    pprint.pprint(vars(args))
