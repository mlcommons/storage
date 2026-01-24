#!/usr/bin/python3.9
#!/usr/bin/env python3
"""
MLPerf Storage Benchmark - Main Entry Point

This module provides the main entry point for the MLPerf Storage
benchmark suite, with comprehensive error handling and user-friendly
messaging.
"""

import signal
import sys
import traceback

from mlpstorage.benchmarks import TrainingBenchmark, VectorDBBenchmark, CheckpointingBenchmark
from mlpstorage.cli_parser import parse_arguments, validate_args, update_args
from mlpstorage.config import HISTFILE, DATETIME_STR, EXIT_CODE, DEFAULT_RESULTS_DIR, get_datetime_string, HYDRA_OUTPUT_SUBDIR
from mlpstorage.debug import debugger_hook, MLPS_DEBUG
from mlpstorage.history import HistoryTracker
from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.report_generator import ReportGenerator
from mlpstorage.errors import (
    MLPStorageException,
    ConfigurationError,
    BenchmarkExecutionError,
    ValidationError,
    FileSystemError,
    MPIError,
    DependencyError,
    ErrorCode,
)
from mlpstorage.error_messages import format_error, ErrorFormatter
from mlpstorage.lockfile import (
    generate_lockfile,
    generate_lockfiles_for_project,
    validate_lockfile,
    format_validation_report,
    LockfileGenerationError,
    GenerationOptions,
)
from mlpstorage.validation_helpers import validate_benchmark_environment

logger = setup_logging("MLPerfStorage")
signal_received = False
error_formatter = ErrorFormatter(use_colors=True)


def signal_handler(sig, frame):
    """Handle signals like SIGINT (Ctrl+C) and SIGTERM."""
    global signal_received

    signal_name = signal.Signals(sig).name
    logger.warning(f"Received signal {signal_name} ({sig})")

    # Set the flag to indicate we've received a signal
    signal_received = True

    # For SIGTERM, exit immediately
    if sig in (signal.SIGTERM, signal.SIGINT):
        logger.info("Exiting due to signal")
        sys.exit(EXIT_CODE.INTERRUPTED)


def handle_lockfile_command(args) -> int:
    """Handle lockfile generate/verify commands.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    if args.lockfile_command == "generate":
        try:
            if args.generate_all:
                # Generate both base and full lockfiles
                logger.info("Generating lockfiles...")
                results = generate_lockfiles_for_project(args.pyproject)
                for name, path in results.items():
                    logger.status(f"Generated {name} lockfile: {path}")
                return EXIT_CODE.SUCCESS
            else:
                # Generate single lockfile
                options = GenerationOptions(
                    output_path=args.output,
                    extras=args.extras,
                    generate_hashes=args.hashes,
                    python_version=args.python_version or "",
                )
                logger.info(f"Generating lockfile: {args.output}")
                _, path = generate_lockfile(args.pyproject, options)
                logger.status(f"Generated lockfile: {path}")
                return EXIT_CODE.SUCCESS
        except LockfileGenerationError as e:
            logger.error(f"Lockfile generation failed: {e}")
            if e.stderr:
                logger.debug(f"stderr: {e.stderr}")
            return EXIT_CODE.FAILURE
        except FileNotFoundError as e:
            logger.error(str(e))
            return EXIT_CODE.FAILURE

    elif args.lockfile_command == "verify":
        try:
            skip = set(args.skip_packages) if args.skip_packages else None
            result = validate_lockfile(
                args.lockfile,
                skip_packages=skip,
                fail_on_missing=not args.allow_missing,
            )

            # Print report
            report = format_validation_report(result)
            if result.valid:
                logger.status(report)
                return EXIT_CODE.SUCCESS
            else:
                logger.error(report)
                return EXIT_CODE.FAILURE
        except FileNotFoundError:
            logger.error(f"Lockfile not found: {args.lockfile}")
            logger.info("Generate a lockfile with: mlpstorage lockfile generate")
            return EXIT_CODE.FAILURE

    return EXIT_CODE.FAILURE


def run_benchmark(args, run_datetime):
    """
    Run a benchmark based on the provided args.

    Args:
        args: Parsed command line arguments.
        run_datetime: Datetime string for this run.

    Returns:
        Exit code indicating success or failure.

    Raises:
        ConfigurationError: If benchmark type is unsupported.
        BenchmarkExecutionError: If benchmark execution fails.
    """
    from mlpstorage.benchmarks import KVCacheBenchmark

    # Validate lockfile if requested
    if hasattr(args, 'verify_lockfile') and args.verify_lockfile:
        logger.info(f"Validating packages against lockfile: {args.verify_lockfile}")
        try:
            result = validate_lockfile(args.verify_lockfile, fail_on_missing=False)
            if not result.valid:
                report = format_validation_report(result)
                logger.error("Package version mismatch detected:")
                logger.error(report)
                logger.error("")
                logger.error("To fix, run one of:")
                logger.error(f"  pip install -r {args.verify_lockfile}")
                logger.error("  uv pip sync " + args.verify_lockfile)
                logger.error("")
                logger.error("Or run without lockfile validation:")
                logger.error(f"  {' '.join(sys.argv).replace('--verify-lockfile ' + args.verify_lockfile, '').strip()}")
                return EXIT_CODE.FAILURE
            logger.status(f"Package validation passed ({result.matched} packages verified)")
        except FileNotFoundError:
            logger.error(f"Lockfile not found: {args.verify_lockfile}")
            logger.error("Generate a lockfile with: mlpstorage lockfile generate")
            return EXIT_CODE.FAILURE

    # Fail-fast environment validation (unless skipped)
    # This validates dependencies, SSH connectivity, paths, etc. BEFORE benchmark instantiation
    skip_validation = getattr(args, 'skip_validation', False)
    if not skip_validation:
        validate_benchmark_environment(args, logger=logger)
    else:
        logger.warning("Skipping environment validation (--skip-validation flag)")

    program_switch_dict = dict(
        training=TrainingBenchmark,
        checkpointing=CheckpointingBenchmark,
        vectordb=VectorDBBenchmark,
        kvcache=KVCacheBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    if not benchmark_class:
        available = list(program_switch_dict.keys())
        raise ConfigurationError(
            f"Unsupported benchmark type: {args.program}",
            parameter="program",
            expected=available,
            actual=args.program,
            suggestion=f"Use one of: {', '.join(available)}",
            code=ErrorCode.CONFIG_INVALID_VALUE
        )

    benchmark = benchmark_class(args, run_datetime=run_datetime, logger=logger)
    ret_code = EXIT_CODE.SUCCESS

    try:
        ret_code = benchmark.run()
    except MLPStorageException:
        # Re-raise our custom exceptions to be handled by main()
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        raise BenchmarkExecutionError(
            f"Benchmark execution failed: {str(e)}",
            exit_code=getattr(e, 'returncode', None),
            suggestion="Check the benchmark logs for details",
            code=ErrorCode.BENCHMARK_COMMAND_FAILED
        ) from e
    finally:
        # Always try to write metadata
        try:
            logger.status(f'Writing metadata for benchmark to: {benchmark.metadata_file_path}')
            benchmark.write_metadata()
        except Exception as e:
            logger.warning(f"Failed to write metadata: {str(e)}")

    return ret_code


def _main_impl():
    """
    Main implementation with error handling.

    This is the actual implementation of main(), separated out
    so that main() can wrap it with exception handling.
    """
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

    if args.program == "lockfile":
        return handle_lockfile_command(args)

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
            logger.warning('Caught signal, exiting...')
            return EXIT_CODE.INTERRUPTED

        ret_code = run_benchmark(args, run_datetime)
        if ret_code != EXIT_CODE.SUCCESS:
            logger.error(f"Benchmark failed after {i+1} iterations")
            return EXIT_CODE.FAILURE

        # Set datetime for next iteration
        run_datetime = get_datetime_string()

    return EXIT_CODE.SUCCESS


def main():
    """
    Main entry point with comprehensive error handling.

    This function wraps _main_impl() to catch and handle all
    exceptions with user-friendly error messages.
    """
    try:
        return _main_impl()

    except ConfigurationError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.CONFIG_ERROR if hasattr(EXIT_CODE, 'CONFIG_ERROR') else EXIT_CODE.FAILURE

    except BenchmarkExecutionError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.ERROR if hasattr(EXIT_CODE, 'ERROR') else EXIT_CODE.FAILURE

    except ValidationError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.FAILURE

    except FileSystemError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.FILE_NOT_FOUND if hasattr(EXIT_CODE, 'FILE_NOT_FOUND') else EXIT_CODE.FAILURE

    except MPIError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.FAILURE

    except DependencyError as e:
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.FAILURE

    except MLPStorageException as e:
        # Catch-all for any other custom exceptions
        logger.error(str(e))
        if e.suggestion:
            logger.info(f"Suggestion: {e.suggestion}")
        return EXIT_CODE.FAILURE

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return EXIT_CODE.INTERRUPTED

    except SystemExit as e:
        # Re-raise SystemExit to allow clean exits
        raise

    except Exception as e:
        # Unexpected exceptions - show full traceback in debug mode
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(format_error('INTERNAL_ERROR', error=str(e)))

        # Show traceback if in debug mode
        if MLPS_DEBUG:
            logger.debug("Stack trace:")
            traceback.print_exc()
        else:
            logger.info("Run with --debug for full stack trace")

        return EXIT_CODE.ERROR if hasattr(EXIT_CODE, 'ERROR') else EXIT_CODE.FAILURE


if __name__ == "__main__":
    sys.exit(main())
