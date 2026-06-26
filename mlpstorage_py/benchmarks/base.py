"""
Base Benchmark Class for MLPerf Storage.

This module provides the abstract base class for all benchmark implementations.
The Benchmark class implements BenchmarkInterface and provides common
functionality including:

- Cluster information collection via MPI
- Result directory management
- Metadata generation and persistence
- Verification/validation integration
- Command execution with signal handling

Classes:
    Benchmark: Abstract base class implementing BenchmarkInterface.

Subclassing:
    To create a new benchmark type:

    1. Inherit from Benchmark
    2. Set BENCHMARK_TYPE class attribute
    3. Implement _run() method
    4. Optionally override generate_command(), validate_args(), etc.

Example:
    class MyBenchmark(Benchmark):
        BENCHMARK_TYPE = BENCHMARK_TYPES.my_benchmark

        def _run(self):
            cmd = self.generate_my_command()
            stdout, stderr, rc = self._execute_command(cmd)
            return rc
"""

import abc
import json
import os
import pprint
import signal
import sys
import time
import types
import uuid
from argparse import Namespace
from typing import Tuple, Dict, Any, List, Optional, Callable, Set, TYPE_CHECKING

from functools import wraps

from mlpstorage_py.config import PARAM_VALIDATION, DATETIME_STR, MLPS_DEBUG, EXEC_TYPE
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.run_directory import (
    DEFAULT_COLLISION_BUMP_BUDGET,
    reserve_run_directory,
)
from mlpstorage_py.debug import debug_tryer_wrapper
from mlpstorage_py.interfaces import BenchmarkInterface, BenchmarkConfig, BenchmarkCommand
from mlpstorage_py.mlps_logging import setup_logging, apply_logging_options
from mlpstorage_py.rules import BenchmarkVerifier, generate_output_location, ClusterInformation
from mlpstorage_py.rules.models import ClusterSnapshots, TimeSeriesData, TimeSeriesSample
from mlpstorage_py.utils import CommandExecutor, MLPSJsonEncoder
from mlpstorage_py.cluster_collector import (
    collect_cluster_info,
    SSHClusterCollector,
    TimeSeriesCollector,
    MultiHostTimeSeriesCollector,
    run_shared_fs_probe,
)
from mlpstorage_py.progress import create_stage_progress, progress_context
from mlpstorage_py.system_description.auto_generator import write_systemname_yaml
from mlpstorage_py.benchmarks.capacity_gate import check_capacity_4field

if TYPE_CHECKING:
    import logging


class Benchmark(BenchmarkInterface, abc.ABC):
    """Base class for all MLPerf Storage benchmarks.

    This abstract class implements BenchmarkInterface and provides common
    functionality for all benchmark types. Subclasses must implement:
    - _run(): The actual benchmark execution logic
    - BENCHMARK_TYPE: Class attribute defining the benchmark type

    The class supports dependency injection for cluster collectors and validators
    to enable easier testing and flexibility.

    Attributes:
        BENCHMARK_TYPE: Class attribute defining the benchmark type enum value.
        args: Parsed command-line arguments.
        logger: Logger instance for output.
        run_datetime: Timestamp string for the run.
        cluster_information: Collected cluster system information.
    """

    BENCHMARK_TYPE = None

    def __init__(
        self,
        args: Namespace,
        logger: Optional['logging.Logger'] = None,
        run_datetime: Optional[str] = None,
        run_number: int = 0,
        cluster_collector: Optional[Any] = None,
        validator: Optional[Any] = None
    ) -> None:
        """Initialize the benchmark.

        Args:
            args: Parsed command-line arguments (argparse.Namespace).
            logger: Optional logger instance. If not provided, one will be created.
            run_datetime: Optional datetime string in YYYYMMDD_HHMMSS format.
                          Defaults to current time.
            run_number: Run number for this benchmark execution (for loops).
            cluster_collector: Optional cluster collector for dependency injection.
                               Used for testing without MPI.
            validator: Optional validator for dependency injection.
                       Used for testing validation logic.
        """
        self.args = args
        # Defense-in-depth (Pitfall 3): the orgname-resolution gate in
        # `main._main_impl` must have populated args.orgname before any
        # Benchmark subclass is instantiated. Production callers never trip
        # this; the guard catches direct (test-only) instantiations or any
        # future codepath that bypasses the main gate.
        if not getattr(self.args, 'orgname', None):
            raise ConfigurationError(
                "orgname was not resolved before Benchmark instantiation",
                suggestion=(
                    "Internal error — orgname must be set on args by "
                    "main._main_impl()'s orgname-resolution gate."
                ),
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        self.debug = self.args.debug or MLPS_DEBUG
        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name=f"{self.BENCHMARK_TYPE}_benchmark", stream_log_level=args.stream_log_level)
            self.logger.warning(f'Benchmark did not get a logger passed. Using default logger.')
            apply_logging_options(self.logger, args)

        if not run_datetime:
            self.logger.warning('No run datetime provided. Using current datetime.')
        self.run_datetime = run_datetime if run_datetime else DATETIME_STR
        self.run_number = run_number
        self.runtime = 0

        # Dependency injection for testability
        self._cluster_collector = cluster_collector
        # Initialize cluster-info attributes up front so the Phase 2
        # systemname.yaml write hook at run() can read them on the
        # early-return path through _collect_cluster_start (which fires
        # for datagen/configview and for any benchmark whose --hosts
        # default is None, e.g. VectorDB). When None, the writer's D-8
        # fallback at auto_generator.py:374-378 takes over via
        # _resolve_host_info_list. See CR-01 in 02-REVIEW.md.
        self._cluster_info_start = None
        # D-43: per-instance sentinel suffix for CAP-02 shared-FS probe
        # (Pitfall 7 collision protection). Generated once per Benchmark
        # instance — NOT per-import or per-module — so concurrent runs
        # against the same data_dir cannot collide on the sentinel path.
        # W-5 launcher contract: this value is passed verbatim to
        # run_shared_fs_probe(...) → mpirun argv[2]; nothing mutates it.
        self._run_uuid = uuid.uuid4().hex
        self._validator = validator

        self.benchmark_run_verifier = None
        self.verification = None
        self.cmd_executor = CommandExecutor(logger=self.logger, debug=args.debug)

        self.command_output_files = list()
        self.run_result_output = self._reserve_run_directory()

        # LAY-06 (Rules.md §2.1.6): capture the live mlpstorage_py/ source
        # tree alongside the results so the submission package is auditable.
        # Per-mode policy:
        #   closed  → ONE image at <rd>/closed/<orgname>/code/ (idempotent).
        #   open    → one image per (benchmark, command) tuple.
        #   whatif  → no image (capture_code_image returns None).
        # WR-09: ``--dry-run`` short-circuits before any work — so we MUST
        # also skip the code-image capture (which would otherwise write
        # ~MBs to disk on every dry-run invocation despite no benchmark
        # actually running). Mirrors the existing ``mode == 'whatif'``
        # no-op inside capture_code_image. ``getattr`` is defensive in
        # case a synthetic args Namespace omits the attribute.
        # Deferred import keeps top-of-file import-time cost minimal and
        # avoids cycles with `mlpstorage_py.results_dir`.
        if getattr(self.args, 'dry_run', False):
            self.code_image_path: Optional[str] = None
        else:
            from mlpstorage_py.results_dir.code_image import capture_code_image
            self.code_image_path: Optional[str] = capture_code_image(
                results_dir=self.args.results_dir,
                mode=self.args.mode,
                orgname=self.args.orgname,
                benchmark_type=self.BENCHMARK_TYPE.name,
                command=getattr(self.args, 'command', 'run'),
            )

        self.metadata_filename = f"{self.BENCHMARK_TYPE.value}_{self.run_datetime}_metadata.json"
        self.metadata_file_path = os.path.join(self.run_result_output, self.metadata_filename)

        # Time-series collection (HOST-04, HOST-05)
        self._timeseries_collector = None
        self._timeseries_data = None
        self.timeseries_filename = f"{self.BENCHMARK_TYPE.value}_{self.run_datetime}_timeseries.json"
        self.timeseries_file_path = os.path.join(self.run_result_output, self.timeseries_filename)

        self.logger.status(f'Benchmark results directory: {self.run_result_output}')

    # =========================================================================
    # BenchmarkInterface Implementation
    # =========================================================================

    @property
    def config(self) -> BenchmarkConfig:
        """Return benchmark configuration.

        Subclasses can override this to provide more specific configuration.
        """
        return BenchmarkConfig(
            name=self.BENCHMARK_TYPE.value if self.BENCHMARK_TYPE else "unknown",
            benchmark_type=self.BENCHMARK_TYPE.name if self.BENCHMARK_TYPE else "unknown",
            supported_commands=self._get_supported_commands(),
            requires_cluster_info=True,
            requires_mpi=getattr(self.args, 'exec_type', None) == EXEC_TYPE.MPI,
        )

    def _get_supported_commands(self) -> List[BenchmarkCommand]:
        """Get list of supported commands. Override in subclass."""
        return [BenchmarkCommand.RUN]

    def validate_args(self, args) -> List[str]:
        """Validate command-line arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            List of error messages. Empty list indicates valid arguments.
        """
        errors = []
        # Subclasses should override to add specific validation
        return errors

    def get_command_handler(self, command: str) -> Optional[Callable]:
        """Return handler function for the given command.

        Args:
            command: Command string (e.g., 'run', 'datagen').

        Returns:
            Callable that handles the command, or None if not supported.
        """
        # Default implementation - subclasses should override
        handlers = {
            'run': self._run,
        }
        return handlers.get(command)

    def generate_command(self, command: str) -> str:
        """Generate the shell command to execute.

        Args:
            command: Command string (e.g., 'run', 'datagen').

        Returns:
            Shell command string ready for execution.
        """
        # Default implementation - subclasses must override for actual command generation
        raise NotImplementedError("Subclasses must implement generate_command()")

    def collect_results(self) -> Dict[str, Any]:
        """Collect and return benchmark results.

        Returns:
            Dictionary containing benchmark results and metadata.
        """
        return {
            'benchmark_type': self.BENCHMARK_TYPE.name if self.BENCHMARK_TYPE else None,
            'run_datetime': self.run_datetime,
            'runtime': self.runtime,
            'verification': self.verification.name if self.verification else None,
            'result_dir': self.run_result_output,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata for recording.

        Returns:
            Dictionary containing benchmark configuration and parameters.
        """
        return self.metadata

    # =========================================================================
    # Original Benchmark Methods
    # =========================================================================

    def _execute_command(
        self,
        command: str,
        output_file_prefix: Optional[str] = None,
        print_stdout: bool = True,
        print_stderr: bool = True
    ) -> Tuple[str, str, int]:
        """Execute the given command and return stdout, stderr, and return code.

        Handles what-if mode, signal watching for graceful termination,
        and optionally saves output to log files.

        Args:
            command: Shell command string to execute.
            output_file_prefix: If provided, stdout/stderr are saved to
                                {prefix}.stdout.log and {prefix}.stderr.log
            print_stdout: Whether to print stdout to console in real-time.
            print_stderr: Whether to print stderr to console in real-time.

        Returns:
            Tuple of (stdout_content, stderr_content, return_code).
            In what-if mode, returns ("", "", 0) without execution.
        """

        self.__dict__.update({'executed_command': command})

        if getattr(self.args, 'dry_run', False) or getattr(self.args, 'what_if', False):
            self.logger.debug(f'Executing command in --dry-run/--what-if mode means no execution will be performed.')
            log_message = f'Dry-run mode: \nCommand: {command}'
            if self.debug:
                log_message += f'\n\nParameters: \n{pprint.pformat(vars(self.args))}'
            self.logger.info(log_message)
            return "", "", 0
        else:
            watch_signals = {signal.SIGINT, signal.SIGTERM}
            stdout, stderr, return_code = self.cmd_executor.execute(command, watch_signals=watch_signals,
                                                                    print_stdout=print_stdout,
                                                                    print_stderr=print_stderr)

            if output_file_prefix:
                stdout_filename = f"{output_file_prefix}.stdout.log"
                stderr_filename = f"{output_file_prefix}.stderr.log"

                stdout_file = os.path.join(self.run_result_output, stdout_filename)
                stderr_file = os.path.join(self.run_result_output, stderr_filename)

                with open(stdout_file, 'w+') as fd:
                    self.logger.verbose(f'Command stdout saved to: {stdout_filename}')
                    fd.write(stdout)

                with open(stderr_file, 'w+') as fd:
                    self.logger.verbose(f'Command stderr saved to: {stderr_filename}')
                    fd.write(stderr)

                self.command_output_files.append(dict(command=command, stdout=stdout_file, stderr=stderr_file))

            return stdout, stderr, return_code

    @staticmethod
    def _apply_dotted_overrides(params, overrides):
        """Merge override_parameters (dotted keys) into a nested params dict.

        Fixes #365: combined_params is frozen at __init__ time from YAML
        defaults + args.params. Subclasses that call add_checkpoint_params()
        afterwards only write into params_dict, leaving combined_params with
        stale YAML defaults. This method folds params_dict back in so that
        metadata['parameters'] reflects the effective run configuration that
        the submission checker reads.
        """
        import copy
        out = copy.deepcopy(params)
        for dotted, value in (overrides or {}).items():
            parts = dotted.split('.')
            cur = out
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        return out

    @property
    def metadata(self) -> Dict[str, Any]:
        """Generate metadata dict capturing the benchmark run configuration.

        This metadata is designed to be complete enough that BenchmarkRunData
        can be reconstructed from it without needing tool-specific result files.

        The metadata includes:
        - benchmark_type, model, command, run_datetime
        - parameters and override_parameters
        - system_info (cluster configuration)
        - runtime, verification status
        - executed_command and output files

        Returns:
            Dictionary containing all benchmark metadata.
        """
        # Core fields required by BenchmarkRunData
        metadata = {
            'benchmark_type': self.BENCHMARK_TYPE.name,
            'model': getattr(self.args, 'model', None),
            'command': getattr(self.args, 'command', None),
            'run_datetime': self.run_datetime,
            'num_processes': getattr(self.args, 'num_processes', None),
            'accelerator': getattr(self.args, 'accelerator_type', None),
            'result_dir': self.run_result_output,
        }

        # Parameters - YAML defaults with CLI overrides folded in (fixes #365).
        # combined_params alone omits overrides added after __init__ (e.g.
        # checkpoint.num_checkpoints_*), causing split-phase runs to double-count.
        if hasattr(self, 'combined_params'):
            metadata['parameters'] = self._apply_dotted_overrides(
                self.combined_params, getattr(self, 'params_dict', {}))
        else:
            metadata['parameters'] = {}

        # Override parameters - user-specified overrides only
        if hasattr(self, 'params_dict'):
            metadata['override_parameters'] = self.params_dict
        else:
            metadata['override_parameters'] = {}

        # System info - serialize ClusterInformation if available
        if hasattr(self, 'cluster_information') and self.cluster_information:
            metadata['system_info'] = self.cluster_information.as_dict()
        else:
            metadata['system_info'] = None

        # Include cluster snapshots if available (start and end collection)
        if hasattr(self, 'cluster_snapshots') and self.cluster_snapshots:
            metadata['cluster_snapshots'] = self.cluster_snapshots.as_dict()

        # Include time-series data reference if available (HOST-04)
        if hasattr(self, '_timeseries_data') and self._timeseries_data:
            metadata['timeseries_data'] = {
                'file': self.timeseries_filename,
                'num_samples': self._timeseries_data.num_samples,
                'interval_seconds': self._timeseries_data.collection_interval_seconds,
                'hosts_collected': self._timeseries_data.hosts_collected,
            }

        # Additional context (not part of BenchmarkRunData but useful)
        metadata['runtime'] = self.runtime
        metadata['verification'] = self.verification.name if self.verification else None
        metadata['executed_command'] = getattr(self, 'executed_command', None)
        metadata['command_output_files'] = self.command_output_files

        # Include full args for debugging/auditing (skip non-serializable)
        try:
            metadata['args'] = vars(self.args)
        except Exception:
            metadata['args'] = str(self.args)

        return metadata

    def write_metadata(self) -> None:
        """Write benchmark metadata to JSON file.

        Writes metadata to {metadata_file_path}. In verbose/debug mode,
        also prints metadata to stdout.
        """
        with open(self.metadata_file_path, 'w+') as fd:
            json.dump(self.metadata, fd, indent=2, cls=MLPSJsonEncoder)

        if self.args.verbose or self.args.debug or self.debug:
            json.dump(self.metadata, sys.stdout, indent=2, cls=MLPSJsonEncoder)

    def write_cluster_info(self):
        """Write detailed cluster information to a separate JSON file."""
        if not hasattr(self, 'cluster_information') or not self.cluster_information:
            return

        cluster_info_filename = f"{self.BENCHMARK_TYPE.value}_cluster_info.json"
        cluster_info_path = os.path.join(self.run_result_output, cluster_info_filename)

        try:
            with open(cluster_info_path, 'w') as fd:
                json.dump(self.cluster_information.to_detailed_dict(), fd, indent=2)
            self.logger.verbose(f'Cluster information saved to: {cluster_info_filename}')
        except Exception as e:
            self.logger.warning(f'Failed to write cluster info: {e}')

    def _should_collect_cluster_info(self) -> bool:
        """Determine if we should collect cluster information via MPI.

        Returns True if:
        - hosts argument is provided and not empty
        - command is not 'datagen' or 'configview' (data generation doesn't need cluster info)
        - skip_cluster_collection is not set
        """
        # Check if hosts are specified
        if not hasattr(self.args, 'hosts') or not self.args.hosts:
            return False

        # Skip for certain commands that don't need cluster info
        if hasattr(self.args, 'command') and self.args.command in ('datagen', 'configview'):
            return False

        # Check if user explicitly disabled collection
        if hasattr(self.args, 'skip_cluster_collection') and self.args.skip_cluster_collection:
            return False

        return True

    def _collect_cluster_information(self) -> 'ClusterInformation':
        """Collect cluster information using MPI if available, otherwise return None.

        This method attempts to collect detailed system information from all hosts
        using MPI. If MPI collection fails or is not available, it returns None
        and the subclass should fall back to CLI args-based collection.

        Returns:
            ClusterInformation instance if collection succeeds, None otherwise.
        """
        if not self._should_collect_cluster_info():
            self.logger.debug('Skipping cluster info collection (conditions not met)')
            return None

        # Only attempt MPI collection if exec_type is MPI
        if not hasattr(self.args, 'exec_type') or self.args.exec_type != EXEC_TYPE.MPI:
            self.logger.debug('Skipping MPI cluster collection (exec_type is not MPI)')
            return None

        try:
            self.logger.debug('Collecting cluster information via MPI...')

            # Get collection parameters
            mpi_bin = getattr(self.args, 'mpi_bin', 'mpirun')
            allow_run_as_root = getattr(self.args, 'allow_run_as_root', False)
            timeout = getattr(self.args, 'cluster_collection_timeout', 60)
            ssh_username = getattr(self.args, 'ssh_username', None)
            shared_staging_dir = getattr(self.args, 'shared_staging_dir', None)

            # Collect cluster info. ``results_dir`` is required by
            # ``collect_cluster_info`` for staging the helper script under
            # ``<results_dir>/collector-staging/`` (see issue #363).
            collected_data = collect_cluster_info(
                hosts=self.args.hosts,
                mpi_bin=mpi_bin,
                logger=self.logger,
                results_dir=self.run_result_output,
                allow_run_as_root=allow_run_as_root,
                timeout_seconds=timeout,
                fallback_to_local=True,
                shared_staging_dir=shared_staging_dir,
                ssh_username=ssh_username,
            )

            # Create ClusterInformation from collected data
            cluster_info = ClusterInformation.from_mpi_collection(collected_data, self.logger)

            # Log collection results
            collection_method = collected_data.get('_metadata', {}).get('collection_method', 'unknown')
            self.logger.debug(
                f'Cluster info collected via {collection_method}: '
                f'{cluster_info.num_hosts} hosts, '
                f'{cluster_info.total_memory_bytes / (1024**3):.1f}GiB total memory, '
                f'{cluster_info.total_cores} total cores'
            )

            # Log any consistency warnings
            if cluster_info.host_consistency_issues:
                for issue in cluster_info.host_consistency_issues:
                    self.logger.warning(f'Cluster consistency: {issue}')

            return cluster_info

        except Exception as e:
            self.logger.warning(f'MPI cluster info collection failed: {e}')
            return None

    def _should_use_ssh_collection(self) -> bool:
        """Determine if SSH-based collection should be used.

        SSH collection is used when:
        - hosts are specified
        - exec_type is NOT MPI (or exec_type is not set)
        - command is 'run' (not datagen/configview)

        Returns:
            True if SSH collection should be used, False otherwise.
        """
        if not hasattr(self.args, 'hosts') or not self.args.hosts:
            return False

        if hasattr(self.args, 'command') and self.args.command in ('datagen', 'configview'):
            return False

        if hasattr(self.args, 'skip_cluster_collection') and self.args.skip_cluster_collection:
            return False

        # Use SSH for non-MPI execution
        if not hasattr(self.args, 'exec_type') or self.args.exec_type != EXEC_TYPE.MPI:
            return True

        return False

    def _collect_via_ssh(self) -> Optional['ClusterInformation']:
        """Collect cluster information using SSH.

        Returns:
            ClusterInformation instance if collection succeeds, None otherwise.
        """
        try:
            self.logger.debug('Collecting cluster information via SSH...')

            ssh_username = getattr(self.args, 'ssh_username', None)
            timeout = getattr(self.args, 'cluster_collection_timeout', 60)

            collector = SSHClusterCollector(
                hosts=self.args.hosts,
                logger=self.logger,
                ssh_username=ssh_username,
                timeout_seconds=timeout
            )

            if not collector.is_available():
                self.logger.warning('SSH not available for cluster collection')
                return None

            result = collector.collect(self.args.hosts, timeout)

            if not result.success:
                self.logger.warning(f'SSH collection had errors: {result.errors}')

            # Create ClusterInformation from collected data
            cluster_info = ClusterInformation.from_mpi_collection(
                {**result.data, '_metadata': {
                    'collection_method': 'ssh',
                    'collection_timestamp': result.timestamp
                }},
                self.logger
            )

            self.logger.debug(
                f'Cluster info collected via SSH: '
                f'{cluster_info.num_hosts} hosts, '
                f'{cluster_info.total_memory_bytes / (1024**3):.1f}GiB total memory'
            )

            return cluster_info

        except Exception as e:
            self.logger.warning(f'SSH cluster info collection failed: {e}')
            return None

    def _collect_cluster_start(self) -> None:
        """Collect cluster information at benchmark start.

        Stores the result in self._cluster_info_start for later use.
        Called at the beginning of run().
        """
        if not self._should_collect_cluster_info() and not self._should_use_ssh_collection():
            self.logger.debug('Skipping start cluster collection (conditions not met)')
            return

        hosts = self.args.hosts if hasattr(self.args, 'hosts') else []
        host_count = len(hosts) if hosts else 1

        self.logger.debug(f"Collecting cluster info ({host_count} host{'s' if host_count != 1 else ''})...")

        with progress_context("Collecting cluster info...", total=None) as (_, set_desc):
            if self._should_use_ssh_collection():
                set_desc("Collecting via SSH...")
                self._cluster_info_start = self._collect_via_ssh()
                self._collection_method = 'ssh'
            else:
                set_desc("Collecting via MPI...")
                self._cluster_info_start = self._collect_cluster_information()
                self._collection_method = 'mpi'

        if self._cluster_info_start:
            self.logger.debug(f'Collected start cluster info via {self._collection_method}')

    def _collect_cluster_end(self) -> None:
        """Collect cluster information at benchmark end.

        Only collects if start collection was performed.
        Creates ClusterSnapshots with both start and end data.
        """
        if not hasattr(self, '_cluster_info_start') or self._cluster_info_start is None:
            self.logger.debug('Skipping end cluster collection (no start collection)')
            return

        self.logger.debug("Collecting end cluster info...")

        with progress_context("Collecting cluster info...", total=None) as (_, set_desc):
            if self._collection_method == 'ssh':
                set_desc("Collecting via SSH...")
                self._cluster_info_end = self._collect_via_ssh()
            else:
                set_desc("Collecting via MPI...")
                self._cluster_info_end = self._collect_cluster_information()

        if self._cluster_info_end:
            self.logger.debug(f'Collected end cluster info via {self._collection_method}')

        # Create ClusterSnapshots
        self.cluster_snapshots = ClusterSnapshots(
            start=self._cluster_info_start,
            end=self._cluster_info_end,
            collection_method=getattr(self, '_collection_method', 'unknown')
        )

        # Also set cluster_information to the start snapshot for backward compatibility
        self.cluster_information = self._cluster_info_start

    def _should_collect_timeseries(self) -> bool:
        """Determine if time-series collection should be performed.

        Returns:
            True if time-series collection should be performed.
        """
        # Check if user explicitly disabled
        if hasattr(self.args, 'skip_timeseries') and self.args.skip_timeseries:
            return False

        # Only collect for 'run' command
        if hasattr(self.args, 'command') and self.args.command not in ('run',):
            return False

        # Skip in dry-run/what-if mode
        if getattr(self.args, 'dry_run', False) or getattr(self.args, 'what_if', False):
            return False

        return True

    def _start_timeseries_collection(self) -> None:
        """Start time-series collection in background.

        Uses MultiHostTimeSeriesCollector if hosts specified,
        otherwise uses single-host TimeSeriesCollector.

        Collection runs in a background thread to minimize performance impact
        on benchmark execution (HOST-05 requirement).
        """
        if not self._should_collect_timeseries():
            self.logger.debug('Skipping time-series collection (disabled or not applicable)')
            return

        interval = getattr(self.args, 'timeseries_interval', 10.0)
        max_samples = getattr(self.args, 'max_timeseries_samples', 3600)

        try:
            if hasattr(self.args, 'hosts') and self.args.hosts:
                # Multi-host collection
                ssh_username = getattr(self.args, 'ssh_username', None)
                ssh_timeout = getattr(self.args, 'cluster_collection_timeout', 30)

                self._timeseries_collector = MultiHostTimeSeriesCollector(
                    hosts=self.args.hosts,
                    interval_seconds=interval,
                    max_samples=max_samples,
                    ssh_username=ssh_username,
                    ssh_timeout=ssh_timeout,
                    logger=self.logger
                )
                self.logger.debug(
                    f'Starting multi-host time-series collection ({len(self.args.hosts)} hosts, '
                    f'interval={interval}s)'
                )
            else:
                # Single-host collection (localhost only)
                self._timeseries_collector = TimeSeriesCollector(
                    interval_seconds=interval,
                    max_samples=max_samples,
                    logger=self.logger
                )
                self.logger.debug(
                    f'Starting single-host time-series collection (interval={interval}s)'
                )

            self._timeseries_collector.start()

        except Exception as e:
            self.logger.warning(f'Failed to start time-series collection: {e}')
            self._timeseries_collector = None

    def _stop_timeseries_collection(self) -> None:
        """Stop time-series collection and store results."""
        if self._timeseries_collector is None:
            return

        try:
            if isinstance(self._timeseries_collector, MultiHostTimeSeriesCollector):
                samples_by_host = self._timeseries_collector.stop()
                hosts_collected = self._timeseries_collector.get_hosts_with_data()

                # Convert to TimeSeriesSample dataclasses
                samples_by_host_typed = {}
                total_samples = 0
                for host, samples in samples_by_host.items():
                    samples_by_host_typed[host] = [
                        TimeSeriesSample.from_dict(s) for s in samples
                    ]
                    total_samples += len(samples)

                self._timeseries_data = TimeSeriesData(
                    collection_interval_seconds=self._timeseries_collector.interval_seconds,
                    start_time=self._timeseries_collector.start_time or '',
                    end_time=self._timeseries_collector.end_time or '',
                    num_samples=total_samples,
                    samples_by_host=samples_by_host_typed,
                    collection_method='ssh' if len(hosts_collected) > 1 else 'local',
                    hosts_requested=list(self._timeseries_collector.hosts),
                    hosts_collected=hosts_collected,
                )

            else:
                # Single-host TimeSeriesCollector
                samples = self._timeseries_collector.stop()
                hostname = samples[0]['hostname'] if samples else 'localhost'

                samples_typed = [TimeSeriesSample.from_dict(s) for s in samples]

                self._timeseries_data = TimeSeriesData(
                    collection_interval_seconds=self._timeseries_collector.interval_seconds,
                    start_time=self._timeseries_collector.start_time or '',
                    end_time=self._timeseries_collector.end_time or '',
                    num_samples=len(samples),
                    samples_by_host={hostname: samples_typed},
                    collection_method='local',
                    hosts_requested=[hostname],
                    hosts_collected=[hostname] if samples else [],
                )

            self.logger.debug(
                f'Time-series collection complete ({self._timeseries_data.num_samples} samples)'
            )

        except Exception as e:
            self.logger.warning(f'Failed to stop time-series collection: {e}')
            self._timeseries_data = None

    def write_timeseries_data(self) -> None:
        """Write time-series data to JSON file.

        Output file follows naming convention: {benchmark_type}_{datetime}_timeseries.json
        This ensures the file is discoverable alongside other benchmark output files
        (HOST-04 requirement).
        """
        if self._timeseries_data is None:
            return

        try:
            with open(self.timeseries_file_path, 'w') as f:
                json.dump(self._timeseries_data.to_dict(), f, indent=2, cls=MLPSJsonEncoder)
            self.logger.verbose(f'Time-series data saved to: {self.timeseries_filename}')
        except Exception as e:
            self.logger.warning(f'Failed to write time-series data: {e}')

    def generate_output_location(self) -> str:
        """Generate the output directory path for this benchmark run.

        Creates a path based on BENCHMARK_TYPE, model, command, and datetime.

        Returns:
            Absolute path string for the result directory.

        Raises:
            ValueError: If BENCHMARK_TYPE is not set.
        """
        if not self.BENCHMARK_TYPE:
            raise ValueError('No benchmark specified. Unable to generate output location')
        # Thread the validated orgname/systemname stashed by
        # capture_or_verify_code_image (code_image.py: args._validated_orgname /
        # args._validated_systemname) so generate_output_location's
        # OPEN/CLOSED ConfigurationError path doesn't fire. For legacy /
        # whatif modes these attrs are absent (getattr default None) and the
        # function's mode check skips the orgname/systemname requirement.
        return generate_output_location(
            self,
            self.run_datetime,
            orgname=getattr(self.args, "_validated_orgname", None),
            systemname=getattr(self.args, "_validated_systemname", None),
        )

    _COLLISION_BUMP_BUDGET = DEFAULT_COLLISION_BUMP_BUDGET

    def _reserve_run_directory(self) -> str:
        """Atomically reserve a unique run directory, updating run_datetime
        if a collision pushes the timestamp forward. See
        mlpstorage_py.benchmarks.run_directory.reserve_run_directory.
        """
        def _path_for(dt: str) -> str:
            self.run_datetime = dt
            return self.generate_output_location()

        reserved, final_dt = reserve_run_directory(
            self.run_datetime, _path_for, budget=self._COLLISION_BUMP_BUDGET
        )
        self.run_datetime = final_dt
        return reserved

    def verify_benchmark(self) -> bool:
        """Verify benchmark parameters meet OPEN or CLOSED requirements.

        Uses BenchmarkVerifier to check if the current configuration
        meets the requirements for closed or open submission.

        Returns:
            True if verification passes, False otherwise.
            May call sys.exit(1) if invalid and --allow-invalid-params not set.
        """
        self.logger.verboser(f'Verifying benchmark parameters: {self.args}')
        if not self.benchmark_run_verifier:
            self.benchmark_run_verifier = BenchmarkVerifier(self, logger=self.logger)

        self.verification = self.benchmark_run_verifier.verify()
        self.logger.verboser(f'Benchmark verification result: {self.verification}')

        # ``args.mode`` is the source of truth (post-PR #412 modal CLI:
        # closed|open|whatif as the first positional, argparse-enforced).
        # The ``not closed_mode and not open_mode`` branch below is the
        # explicit short-circuit for ``mode == 'whatif'``: whatif runs
        # without verification by design. Do not delete it — whatif is a
        # supported execution mode.
        mode = self.args.mode
        closed_mode = (mode == 'closed')
        open_mode = (mode == 'open')

        if not closed_mode and not open_mode:
            self.logger.warning(f'Running the benchmark without verification for open or closed configurations. These results are not valid for submission. Use closed or open as the first positional argument to specify a configuration.')
            return True
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to verify benchmark')

        if not self.verification:
            self.logger.error(f'Verification did not return a result. Contact the developer')
            sys.exit(1)
        if self.verification == PARAM_VALIDATION.CLOSED:
            return True
        elif self.verification == PARAM_VALIDATION.INVALID:
            if self.args.allow_invalid_params:
                self.logger.warning(f'Invalid configuration found. Allowing the benchmark to proceed.')
                return True
            else:
                self.logger.error(f'Invalid configuration found. Aborting benchmark run.')
                sys.exit(1)

        if self.verification == PARAM_VALIDATION.OPEN:
            if open_mode:
                self.logger.status(f'Running as allowed open configuration')
                return True
            else:
                # closed_mode is True here
                self.logger.warning(f'Parameters allowed for open but not closed. Use --open and rerun the benchmark.')
                sys.exit(1)

    def required_bytes_for_capacity_gate(self) -> int:
        """Return bytes needed for the dataset destination (CAP-01).

        Subclasses MUST override. Each benchmark's required-bytes math
        already lives inline in its ``datasize``/``execute_datasize``
        method (per-benchmark — see 05-RESEARCH.md §"Per-benchmark
        required_bytes sources (CAP-01)"). The override mirrors that math
        WITHOUT calling the user-facing logger so the happy path can stay
        silent per REQUIREMENTS.md SC#6.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override "
            f"required_bytes_for_capacity_gate() for CAP-01"
        )

    def _capacity_gate_destination(self) -> Optional[str]:
        """Return the filesystem path the gate runs statvfs against (CAP-01).

        Return ``None`` to skip the local statvfs — used by the A8 remote-
        backend escape hatch (e.g., VectorDB pointed at a remote milvus
        URI). The skip is logged at INFO level by ``_pre_execution_gate``.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override "
            f"_capacity_gate_destination() for CAP-01"
        )

    def _pre_execution_gate(self) -> None:
        """Run all pre-execution capacity/environment gates (CAP-01 in
        Slice 3; CAP-02 shared-FS verification is appended in Slice 4 of
        Phase 5; LIFE-02 stays on the run-only path via the existing
        ``write_systemname_yaml`` try/except).

        Called from BOTH ``Benchmark.run()`` AND each subclass's datagen
        entry point (TrainingBenchmark.datasize, CheckpointingBenchmark.
        datasize, VectorDBBenchmark.execute_datagen, KVCacheBenchmark's
        kvcache datagen branch in ``_run``). Per REQUIREMENTS.md CAP-01 +
        RESEARCH Pitfall 5 the gate runs per-rank so a single starved node
        in a heterogeneous fleet fails fast with its own destination
        identified in the error.

        Happy path: returns ``None`` silently (no logger output) per SC#6.
        """
        destination = self._capacity_gate_destination()
        if destination is None:
            # A8 escape hatch: remote vector-DB backend (milvus/elasticsearch/
            # pgvector), or any other engine whose data lands behind a network
            # boundary where local statvfs is meaningless. Log INFO so the
            # operator sees the skip; do not raise.
            self.logger.info(
                "CAP-01 skipped: destination not local "
                "(e.g., remote vector-DB backend)"
            )
            return
        required_bytes = self.required_bytes_for_capacity_gate()
        check_capacity_4field(destination, required_bytes, self.logger)
        # ------------------------------------------------------------------
        # Slice 4 / CAP-02: shared-FS verification (Phase 5 / Plan 05-04).
        # ------------------------------------------------------------------
        # On multi-host runs, verify the data-dir is the SAME shared
        # filesystem on every participating host (REQUIREMENTS.md CAP-02).
        # The probe is a silent no-op on single-host runs (SC#8). The
        # `self._run_uuid` is the Pitfall-7 per-instance UUID, generated
        # once in __init__ and passed through to mpirun argv verbatim
        # (W-5 launcher pass-through contract). The destination reused
        # here is the same path CAP-01 just statvfs'd.
        hosts = getattr(self.args, 'hosts', None) or []
        run_shared_fs_probe(
            destination=destination,
            hosts=hosts,
            run_uuid=self._run_uuid,
            logger=self.logger,
            mpi_bin=getattr(self.args, 'mpi_bin', None),
            allow_run_as_root=getattr(self.args, 'allow_run_as_root', False),
            ssh_username=getattr(self.args, 'ssh_username', None),
        )

    @abc.abstractmethod
    def _run(self) -> int:
        """Run the actual benchmark execution.

        Subclasses must implement this method to define the benchmark
        execution logic. The method should:

        1. Generate and execute the benchmark command
        2. Collect and process results
        3. Write metadata and output files
        4. Return the exit code

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        raise NotImplementedError

    def _validate_environment(self) -> None:
        """Validate environment before benchmark execution.

        Called early in run() to catch configuration issues before
        any work is done. Subclasses can override to add benchmark-
        specific validation.

        Note: Primary environment validation is done in main.py via
        validate_benchmark_environment() BEFORE benchmark instantiation.
        This hook is for benchmark-specific validation that requires
        the benchmark instance to exist.

        Raises:
            DependencyError: If required dependencies are missing.
            ConfigurationError: If configuration is invalid.
        """
        # Environment validation is primarily done in main.py before
        # benchmark instantiation. This hook allows subclasses to add
        # benchmark-specific validation if needed.
        pass

    def run(self) -> int:
        """Execute the benchmark and track runtime.

        Wraps _run() with timing measurement, cluster collection, and
        time-series collection. Shows stage indicators during execution.

        Collects cluster information at start and end of benchmark
        (HOST-03 requirement).

        Collects time-series data during benchmark execution using a
        background thread to minimize performance impact (HOST-04, HOST-05).

        Returns:
            Exit code from _run().
        """
        stages = [
            "Validating environment...",
            "Collecting cluster info...",
            "Running benchmark...",
            "Processing results...",
        ]

        with create_stage_progress(stages, logger=self.logger) as advance_stage:
            # Stage 1: Validation
            self._validate_environment()
            advance_stage()

            # Stage 2: Cluster collection
            self._collect_cluster_start()

            # Phase 5 CAP-01 pre-execution gate (Slice 3 of Phase 5; Slice 4
            # extends the body with CAP-02 shared-FS probe). Fires AFTER
            # cluster collection so a per-rank destination check has the
            # cluster context available, BEFORE write_systemname_yaml so a
            # starved disk fails fast BEFORE any on-disk artifact is created.
            self._pre_execution_gate()

            # Phase 2 LIFE-01 write hook. Fires AFTER cluster collection so the
            # write can consume self._cluster_info_start; BEFORE DLIO launch so
            # the file lands before any benchmark output exists. The writer owns
            # its own args.command == 'run' gate (D-12 — belt-and-braces with
            # _should_collect_cluster_info()).
            try:
                write_systemname_yaml(self.args, self._cluster_info_start, self.logger)
            except FileExistsError:
                # D-9 no-op-if-exists is handled INSIDE write_systemname_yaml
                # (the function returns None). Any FileExistsError that bubbles
                # up here is unexpected; re-raise rather than swallow.
                raise
            except Exception as e:
                # D-9: filesystem failures (EACCES, ENOSPC, IsADirectoryError, etc.)
                # abort the benchmark BEFORE DLIO launches. The universal
                # collection-failure rule applies to COLLECTOR failures (which
                # yield empty strings), NOT to filesystem-level WRITE failures.
                self.logger.error(f"Failed to write systemname.yaml: {e}")
                raise

            self._start_timeseries_collection()
            advance_stage()

            # Stage 3: Benchmark execution
            # Note: Stage progress remains visible showing elapsed time
            # during this phase. DLIO output flows through directly.
            start_time = time.time()
            try:
                result = self._run()
            finally:
                self.runtime = time.time() - start_time
                advance_stage()

                # Stage 4: Cleanup/Processing
                self._stop_timeseries_collection()
                self._collect_cluster_end()
                self.write_timeseries_data()
                advance_stage()

        return result



