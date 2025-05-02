#!/usr/bin/python3.9
#!/usr/bin/env python3
import sys

from mlpstorage.benchmarks import TrainingBenchmark, VectorDBBenchmark
from mlpstorage.cli import parse_arguments, validate_args, update_args
from mlpstorage.config import HISTFILE, DATETIME_STR, EXIT_CODE
from mlpstorage.history import HistoryTracker
from mlpstorage.logging import setup_logging, apply_logging_options

logger = setup_logging("MLPerfStorage")


def run_benchmark(args):
    """Run a benchmark based on the provided args."""
    validate_args(args)
    update_args(args)
    program_switch_dict = dict(
        training=TrainingBenchmark,
        vectordb=VectorDBBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    if not benchmark_class:
        print(f"Unsupported program: {args.program}")
        return 1
        
    benchmark = benchmark_class(args)
    ret_code = benchmark.run()
    benchmark.write_metadata()
    return ret_code


def main():
    args = parse_arguments()
    apply_logging_options(logger, args)

    hist = HistoryTracker(history_file=HISTFILE, logger=logger)
    if args.program != "history":
        # Don't save history commands
        hist.add_entry(sys.argv)

    # Handle history command separately
    if args.program == 'history':
        new_args = hist.handle_history_command(args)

        # Check if we got new args back (not just an exit code)
        if isinstance(new_args, EXIT_CODE):
            # We got an exit code, so return it
            return new_args

        elif isinstance(new_args, object) and hasattr(new_args, 'program'):
            # Check if logging options have changed
            if (hasattr(new_args, 'debug') and new_args.debug != args.debug) or \
               (hasattr(new_args, 'verbose') and new_args.verbose != args.verbose) or \
               (hasattr(new_args, 'stream_log_level') and new_args.stream_log_level != args.stream_log_level):
                # Apply the new logging options
                apply_logging_options(logger, new_args)
            
            args = new_args
        else:
            # If handle_history_command returned an exit code, return it
            return new_args
    
    # For other commands, run the benchmark
    return run_benchmark(args)

if __name__ == "__main__":
    sys.exit(main())
