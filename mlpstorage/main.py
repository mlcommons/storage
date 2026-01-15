#!/usr/bin/python3.9
#!/usr/bin/env python3
import signal
import sys

from mlpstorage.benchmarks import TrainingBenchmark, VectorDBBenchmark, CheckpointingBenchmark
from mlpstorage.cli_parser import parse_arguments, validate_args, update_args
from mlpstorage.config import HISTFILE, DATETIME_STR, EXIT_CODE, DEFAULT_RESULTS_DIR, get_datetime_string, HYDRA_OUTPUT_SUBDIR
from mlpstorage.debug import debugger_hook, MLPS_DEBUG
from mlpstorage.history import HistoryTracker
from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.reporting import ReportGenerator

logger = setup_logging("MLPerfStorage")
signal_received = False


def signal_handler(sig, frame):
    """Handle signals like SIGINT (Ctrl+C) and SIGTERM."""
    global signal_received

    signal_name = signal.Signals(sig).name
    logger.warning(f"Received signal {signal_name} ({sig})")

    # Set the flag to indicate we've received a signal
    signal_received = True

    # For SIGTERM, exit immediately
    if sig in (signal.SIGTERM, signal.SIGINT):
        logger.info("Exiting immediately due to SIGTERM")
        sys.exit(EXIT_CODE.INTERRUPTED)


def run_benchmark(args, run_datetime):
    """Run a benchmark based on the provided args."""
    program_switch_dict = dict(
        training=TrainingBenchmark,
        checkpointing=CheckpointingBenchmark,
        vectordb=VectorDBBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    if not benchmark_class:
        print(f"Unsupported program: {args.program}")
        return 1
        
    benchmark = benchmark_class(args, run_datetime=run_datetime, logger=logger)
    try:
        ret_code = benchmark.run()
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        ret_code = EXIT_CODE.ERROR
    finally:
        logger.status(f'Writing metadata for benchmark to: {benchmark.metadata_file_path}')
        try:
            benchmark.write_metadata()
            return ret_code
        except Exception as e:
            logger.error(f"Error writing metadata: {str(e)}")
            return ret_code


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    global signal_received

    args = parse_arguments()
    if args.debug or MLPS_DEBUG:
        sys.excepthook = debugger_hook

    apply_logging_options(logger, args)

    datetime_str = DATETIME_STR

    hist = HistoryTracker(history_file=HISTFILE, logger=logger)
    if args.program != "history":
        # Don't save history commands
        hist.add_entry(sys.argv, datetime_str=datetime_str)

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

    if args.program == "reports":
        results_dir = args.results_dir if hasattr(args, 'results_dir') else DEFAULT_RESULTS_DIR
        report_generator = ReportGenerator(results_dir, args, logger=logger)
        return report_generator.generate_reports()

    run_datetime = datetime_str

    # Handle vdb end conditions, num_process standardization, and args.params flattening
    update_args(args)

    # For other commands, run the benchmark
    for i in range(args.loops):
        if signal_received:
            print(f'Caught signal, exiting...')
            return EXIT_CODE.INTERRUPTED

        ret_code = run_benchmark(args, run_datetime)
        if ret_code != EXIT_CODE.SUCCESS:
            logger.error(f"Benchmark failed after {i+1} iterations")
            return EXIT_CODE.FAILURE

        # Set datetime for next iteration
        run_datetime = get_datetime_string()

    return EXIT_CODE.SUCCESS

if __name__ == "__main__":
    sys.exit(main())
