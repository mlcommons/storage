#!/usr/bin/python3.9
#!/usr/bin/env python3
"""
MLPerf Storage Benchmark - Main Entry Point

This module provides the main entry point for the MLPerf Storage
benchmark suite, with comprehensive error handling and user-friendly
messaging.
"""

import os
import re
import signal
import sys
import traceback

from mlpstorage_py.cli_parser import parse_arguments, validate_args, update_args
from mlpstorage_py.config import HISTFILE, DATETIME_STR, EXIT_CODE, DEFAULT_RESULTS_DIR, get_datetime_string, HYDRA_OUTPUT_SUBDIR
from mlpstorage_py.debug import debugger_hook, MLPS_DEBUG
from mlpstorage_py.history import HistoryTracker
from mlpstorage_py.mlps_logging import setup_logging, apply_logging_options
from mlpstorage_py.errors import (
    MLPStorageException,
    ConfigurationError,
    BenchmarkExecutionError,
    ValidationError,
    FileSystemError,
    MPIError,
    DependencyError,
    ErrorCode,
)
from mlpstorage_py.error_messages import format_error, ErrorFormatter
from mlpstorage_py.lockfile import (
    generate_lockfile,
    generate_lockfiles_for_project,
    validate_lockfile,
    format_validation_report,
    LockfileGenerationError,
    GenerationOptions,
)
from mlpstorage_py.validation_helpers import validate_benchmark_environment
from mlpstorage_py.progress import progress_context
from mlpstorage_py.results_dir import resolve_orgname
from mlpstorage_py.results_dir.errors import ResultsDirNotInitializedError
from mlpstorage_py.submission_checker.tools.code_image import capture_or_verify_code_image, CodeImageError

logger = setup_logging("MLPerfStorage")
signal_received = False
error_formatter = ErrorFormatter(use_colors=True)

# CONTEXT.md D-12 — modes that DO NOT require a sentinel-resolved orgname.
# Every other mode (closed, open, whatif, reports, history, validate) is
# subject to the LAY-03 orgname-resolution gate in `_main_impl`.
NON_BENCHMARK_NO_ORGNAME_MODES = frozenset({"init", "version", "lockfile", "rules-coverage"})


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
            with progress_context(
                "Generating lockfile...",
                total=None,
                logger=logger
            ) as (update, set_desc):
                if args.generate_all:
                    # Generate both base and full lockfiles
                    set_desc("Generating lockfiles...")
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
                    set_desc(f"Generating lockfile: {args.output}")
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
        with progress_context(
            "Verifying lockfile...",
            total=None,
            logger=logger
        ) as (update, set_desc):
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
    # Lazy-load benchmark classes so that non-benchmark subcommands
    # (validate, rules-coverage, version, lockfile) do not pay the import
    # cost of pyarrow / pymilvus / etc. — letting `mlpstorage validate`
    # run on a base install without the `[full]` extra.
    from mlpstorage_py.benchmarks import (
        TrainingBenchmark,
        VectorDBBenchmark,
        CheckpointingBenchmark,
        KVCacheBenchmark,
    )

    # Validate lockfile if requested
    if hasattr(args, 'verify_lockfile') and args.verify_lockfile:
        with progress_context(
            "Validating packages against lockfile...",
            total=None,
            logger=logger
        ) as (update, set_desc):
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
        with progress_context(
            "Validating environment...",
            total=None,
            logger=logger
        ) as (update, set_desc):
            # Errors from validation will propagate after progress cleanup
            validate_benchmark_environment(args, logger=logger)
    else:
        logger.warning("Skipping environment validation (--skip-validation flag)")

    # Capture/verify code image BEFORE benchmark instantiation (Phase 2 D-07).
    # Helper internally gates on (args.mode, args.command) per D-10, so it is
    # safe to call unconditionally — non-result-generating commands no-op.
    # Helper also owns ALL env-var reading and validation (POSIX regex + inline
    # `.`/`..` path-traversal guard) — see Plan 02 REVIEWS.md consensus finding.
    with progress_context(
        "Capturing or verifying code image...",
        total=None,
        logger=logger
    ) as (update, set_desc):
        capture_or_verify_code_image(args, os.environ, logger)

    program_switch_dict = dict(
        training=TrainingBenchmark,
        checkpointing=CheckpointingBenchmark,
        vectordb=VectorDBBenchmark,
        kvcache=KVCacheBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.benchmark)
    if not benchmark_class:
        available = list(program_switch_dict.keys())
        raise ConfigurationError(
            f"Unsupported benchmark type: {args.benchmark}",
            parameter="benchmark",
            expected=available,
            actual=args.benchmark,
            suggestion=f"Use one of: {', '.join(available)}",
            code=ErrorCode.CONFIG_INVALID_VALUE
        )

    benchmark = benchmark_class(args, run_datetime=run_datetime, logger=logger)

    # Warn if the user is relying on the temp-dir default for results.
    # Results stored in /tmp (or equivalent) are wiped on reboot.
    _results_dir = getattr(args, 'results_dir', DEFAULT_RESULTS_DIR)
    if _results_dir == DEFAULT_RESULTS_DIR and not os.environ.get('MLPERF_RESULTS_DIR'):
        logger.warning(
            f"Results directory not specified. Writing results to the system temp directory: "
            f"{DEFAULT_RESULTS_DIR}. These results will NOT persist across a reboot. "
            f"Use --results-dir <path> or set the MLPERF_RESULTS_DIR environment variable "
            f"to save results permanently."
        )

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

    # CR-03: logging setup is universal — only **benchmark** plumbing is
    # gated for ``init`` (RESEARCH.md Pitfall 3). Calling
    # ``apply_logging_options`` BEFORE the init early-return makes the
    # init dispatcher's ``logger.info(...)`` confirmations visible and
    # lets ``--debug`` triage init failures. The function is defensive
    # via ``hasattr`` checks, so missing ``debug``/``verbose``/
    # ``stream_log_level`` attributes on the init Namespace are fine.
    if getattr(args, 'debug', False) or MLPS_DEBUG:
        sys.excepthook = debugger_hook

    apply_logging_options(logger, args)

    # `init` is a filesystem-local utility that must NOT flow through
    # update_args / validate_benchmark_environment / run_benchmark
    # (RESEARCH.md Pitfall 3). Early-return BEFORE any benchmark plumbing
    # — but AFTER apply_logging_options above so the dispatcher's
    # confirmations are visible.
    if args.mode == "init":
        from mlpstorage_py.results_dir.init import run_init
        return run_init(args)

    if args.mode == "version":
        from mlpstorage_py import VERSION
        print(VERSION)
        sys.exit(0)

    datetime_str = DATETIME_STR

    hist = HistoryTracker(history_file=HISTFILE, logger=logger)
    if args.mode != "history":
        # Don't save history commands
        hist.add_entry(sys.argv, datetime_str=datetime_str)

    # Bypass dispatch for utility modes that do NOT consume an orgname-pinned
    # results-dir. Per CONTEXT.md D-12 the bypass list is exactly four modes:
    # `init` (handled above), `version` (handled above), `lockfile`, and
    # `rules-coverage`. All other modes (closed/open/whatif/reports/history/
    # validate) flow through the LAY-03 orgname-resolution gate below.
    if args.mode == "lockfile":
        return handle_lockfile_command(args)

    if args.mode == "rules-coverage":
        from mlpstorage_py.submission_checker.tools.rules_coverage import run as run_rules_coverage
        return run_rules_coverage(args)

    # ------------------------------------------------------------------ #
    # LAY-03 orgname-resolution gate.
    #
    # Every gated mode that supplies `--results-dir` must have a valid
    # `mlperf-results.yaml` sentinel in that directory; the gate reads the
    # sentinel, pins `args.orgname` for downstream consumers (path generator,
    # banner, benchmark base), and fails fast with the EXACT CONTEXT.md
    # message when the sentinel is missing or malformed.
    #
    # The error message backticks are LOCKED VERBATIM per CONTEXT.md LAY-03
    # / ROADMAP success criterion #2 — do NOT switch to single quotes (`!r`
    # would render `'…'`) and do NOT add backslash escapes (the backtick is
    # not a Python escape sequence; `\\`` produces `\` + backtick on screen).
    # ------------------------------------------------------------------ #
    if args.mode not in NON_BENCHMARK_NO_ORGNAME_MODES:
        results_dir_value = getattr(args, 'results_dir', None)
        if results_dir_value:
            try:
                args.orgname = resolve_orgname(results_dir_value)
            except ResultsDirNotInitializedError as e:
                raise ConfigurationError(
                    f"results-dir `{results_dir_value}` has not been initialized.",
                    suggestion=f"Run `mlpstorage init <orgname> {results_dir_value}` first.",
                    code=ErrorCode.CONFIG_MISSING_REQUIRED,
                ) from e
            # WR-06: defense-in-depth re-validation at the gate. The schema
            # (Pydantic v2) already full-match-validates the sentinel's
            # ``orgname`` against ``[A-Za-z0-9._-]+`` on read, so a properly-
            # written sentinel cannot reach here with a path separator or
            # other unsafe character. But ``args.orgname`` lands in
            # ``os.path.join(..., orgname, "results", ...)`` immediately
            # downstream, so we want a paranoid post-resolution assertion
            # that catches:
            #   * a future Pydantic-version bump that regresses to
            #     non-anchored regex semantics,
            #   * a switch to v1's ``regex=`` keyword (which used different
            #     match semantics),
            #   * any unit-test path that constructs args.orgname directly
            #     without going through ``read_sentinel``.
            if not re.fullmatch(r"[A-Za-z0-9._-]+", args.orgname):
                raise ConfigurationError(
                    f"sentinel orgname {args.orgname!r} contains invalid characters",
                    suggestion=(
                        "Re-initialize the results-dir with a clean orgname "
                        "(matches [A-Za-z0-9._-]+)."
                    ),
                    code=ErrorCode.CONFIG_INVALID_VALUE,
                )

    # Handle history command separately (now AFTER the gate so it inherits
    # a resolved args.orgname when --results-dir is supplied; D-12).
    if args.mode == 'history':
        new_args = hist.handle_history_command(args)

        # Check if we got new args back (not just an exit code)
        if isinstance(new_args, EXIT_CODE):
            # We got an exit code, so return it
            return new_args

        elif isinstance(new_args, object) and hasattr(new_args, 'mode'):
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

        # WR-03: the replayed entry may carry a bypass mode (version /
        # lockfile / rules-coverage) or even 'init'. The bypass dispatches
        # above ran on the ORIGINAL args (mode='history'), so they did not
        # match. Re-route here before falling through to the benchmark
        # loop. Without this, a replayed bypass-mode entry would land in
        # ``update_args`` → ``run_benchmark`` with a non-benchmark mode.
        if args.mode == "init":
            from mlpstorage_py.results_dir.init import run_init
            return run_init(args)
        if args.mode == "version":
            from mlpstorage_py import VERSION
            print(VERSION)
            sys.exit(0)
        if args.mode == "lockfile":
            return handle_lockfile_command(args)
        if args.mode == "rules-coverage":
            from mlpstorage_py.submission_checker.tools.rules_coverage import run as run_rules_coverage
            return run_rules_coverage(args)

    if args.mode == "reports":
        # Lazy-import: ReportGenerator pulls psutil, which is only required
        # for the reports subcommand. Keeping the import here lets validate /
        # rules-coverage / version run on a base install.
        from mlpstorage_py.report_generator import ReportGenerator
        results_dir = args.results_dir if hasattr(args, 'results_dir') else DEFAULT_RESULTS_DIR
        report_generator = ReportGenerator(results_dir, args, logger=logger)
        return report_generator.generate_reports()

    if args.mode == "validate":
        from mlpstorage_py.submission_checker.main import run as run_submission_checker
        return run_submission_checker(args)

    run_datetime = datetime_str

    # Handle vdb end conditions, num_process standardization, and args.params flattening
    update_args(args)

    if not getattr(args, 'quiet', False):
        from mlpstorage_py.run_summary import print_run_summary
        print_run_summary(args)

    # For other commands, run the benchmark
    for i in range(getattr(args, 'loops', 1)):
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

    except CodeImageError as e:
        # Phase 2 D-22: code-image capture/verify failures (incl. MissingHashFile,
        # MalformedHashFile, hash-mismatch CodeImageError) map to a dedicated
        # exit code distinct from generic FAILURE so CI/scripts can detect them.
        # CodeImageError is NOT a MLPStorageException subclass, so it requires
        # an explicit handler ordered BEFORE the MLPStorageException catch-all.
        logger.error(str(e))
        return EXIT_CODE.CODE_IMAGE_ERROR

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

        # Show traceback if in debug mode. MLPS_DEBUG is the env-var path
        # (read at import time); also check `--debug` directly via sys.argv
        # so the CLI flag emits a trace even though `args` is not in scope
        # here. `--debug` is store_true so a bare-token check suffices.
        debug_cli = '--debug' in sys.argv
        if MLPS_DEBUG or debug_cli:
            logger.debug("Stack trace:")
            traceback.print_exc()
        else:
            logger.info("Run with --debug for full stack trace")

        return EXIT_CODE.ERROR if hasattr(EXIT_CODE, 'ERROR') else EXIT_CODE.FAILURE


if __name__ == "__main__":
    sys.exit(main())
