"""
CLI argument parsing for MLPerf Storage benchmarks.

This module provides the main argument parsing entry point,
using modular argument builders from the cli package.
"""

import argparse
import sys

from mlpstorage import VERSION
from mlpstorage.config import LLM_MODELS, VECTORDB_DEFAULT_RUNTIME, EXIT_CODE

# Import modular argument builders from cli package
from mlpstorage.cli import (
    HELP_MESSAGES,
    PROGRAM_DESCRIPTIONS,
    add_universal_arguments,
    add_training_arguments,
    add_checkpointing_arguments,
    add_vectordb_arguments,
    add_reports_arguments,
    add_history_arguments,
)

# Backwards compatibility aliases
help_messages = HELP_MESSAGES
prog_descriptions = PROGRAM_DESCRIPTIONS

def parse_arguments():
    """Parse command-line arguments for MLPerf Storage benchmarks.

    Returns:
        argparse.Namespace: Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(description="Script to launch the MLPerf Storage benchmark")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub_programs = parser.add_subparsers(dest="program", required=True)
    sub_programs.required = True

    # Create subparsers for each benchmark type
    training_parsers = sub_programs.add_parser(
        "training",
        description=PROGRAM_DESCRIPTIONS['training'],
        help="Training benchmark options"
    )
    checkpointing_parsers = sub_programs.add_parser(
        "checkpointing",
        description=PROGRAM_DESCRIPTIONS['checkpointing'],
        help="Checkpointing benchmark options",
        formatter_class=argparse.RawTextHelpFormatter
    )
    vectordb_parsers = sub_programs.add_parser(
        "vectordb",
        description=PROGRAM_DESCRIPTIONS['vectordb'],
        help="VectorDB benchmark options"
    )
    reports_parsers = sub_programs.add_parser(
        "reports",
        description=PROGRAM_DESCRIPTIONS.get('reports', ''),
        help="Generate a report from benchmark results"
    )
    history_parsers = sub_programs.add_parser(
        "history",
        description=PROGRAM_DESCRIPTIONS.get('history', ''),
        help="Display benchmark history"
    )

    sub_programs_map = {
        'training': training_parsers,
        'checkpointing': checkpointing_parsers,
        'vectordb': vectordb_parsers,
        'reports': reports_parsers,
        'history': history_parsers,
    }

    # Add arguments using modular builders from cli package
    add_training_arguments(training_parsers)
    add_checkpointing_arguments(checkpointing_parsers)
    add_vectordb_arguments(vectordb_parsers)
    add_reports_arguments(reports_parsers)
    add_history_arguments(history_parsers)

    # Universal arguments are added within each argument builder now
    # (except for top-level parsers that need them)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if len(sys.argv) == 2 and sys.argv[1] in sub_programs_map.keys():
        sub_programs_map[sys.argv[1]].print_help(sys.stderr)
        sys.exit(1)

    parsed_args = parser.parse_args()
    
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
    error_messages = []
    # Add generic validations here. Workload specific validation is in the Benchmark classes
    if args.program == "checkpointing":
        if args.model not in LLM_MODELS:
            error_messages.append("Invalid LLM model. Supported models are: {}".format(", ".join(LLM_MODELS)))
        if args.num_checkpoints_read < 0 or args.num_checkpoints_write < 0:
            error_messages.append("Number of checkpoints read and write must be non-negative")

    if error_messages:
        for msg in error_messages:
            print(msg)

        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)


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
    if args.params:
        flattened_params = [item for sublist in args.params for item in sublist]
        setattr(args, 'params', flattened_params)

    if args.mpi_params:
        flattened_mpi_params = [item for sublist in args.mpi_params for item in sublist]
        setattr(args,'mpi_params', flattened_mpi_params)

    if hasattr(args, 'hosts'):
        print(f'Hosts is: {args.hosts}')
        # hosts can be comma separated string or a list of strings. If it's a string, it is still a list of length 1
        if len(args.hosts) == 1 and isinstance(args.hosts[0], str):
            setattr(args, 'hosts', args.hosts[0].split(','))
        print(f'Hosts is: {args.hosts}')

    if not hasattr(args, "num_client_hosts") and hasattr(args, "hosts"):
        setattr(args, "num_client_hosts", len(args.hosts))


if __name__ == "__main__":
    args = parse_arguments()
    import pprint
    pprint.pprint(vars(args))

