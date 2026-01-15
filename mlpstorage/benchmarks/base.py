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
from argparse import Namespace
from typing import Tuple, Dict, Any, List, Optional, Callable, Set, TYPE_CHECKING

from functools import wraps

from pyarrow.ipc import open_stream

from mlpstorage.config import PARAM_VALIDATION, DATETIME_STR, MLPS_DEBUG, EXEC_TYPE
from mlpstorage.debug import debug_tryer_wrapper
from mlpstorage.interfaces import BenchmarkInterface, BenchmarkConfig, BenchmarkCommand
from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.rules import BenchmarkVerifier, generate_output_location, ClusterInformation
from mlpstorage.utils import CommandExecutor, MLPSJsonEncoder
from mlpstorage.cluster_collector import collect_cluster_info

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
        self._validator = validator

        self.benchmark_run_verifier = None
        self.verification = None
        self.cmd_executor = CommandExecutor(logger=self.logger, debug=args.debug)

        self.command_output_files = list()
        self.run_result_output = self.generate_output_location()
        os.makedirs(self.run_result_output, exist_ok=True)

        self.metadata_filename = f"{self.BENCHMARK_TYPE.value}_{self.run_datetime}_metadata.json"
        self.metadata_file_path = os.path.join(self.run_result_output, self.metadata_filename)

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

        if self.args.what_if:
            self.logger.debug(f'Executing command in --what-if mode means no execution will be performed.')
            log_message = f'What-if mode: \nCommand: {command}'
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

        # Parameters - prefer combined_params if available (includes YAML + overrides)
        if hasattr(self, 'combined_params'):
            metadata['parameters'] = self.combined_params
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

            # Collect cluster info
            collected_data = collect_cluster_info(
                hosts=self.args.hosts,
                mpi_bin=mpi_bin,
                logger=self.logger,
                allow_run_as_root=allow_run_as_root,
                timeout_seconds=timeout,
                fallback_to_local=True
            )

            # Create ClusterInformation from collected data
            cluster_info = ClusterInformation.from_mpi_collection(collected_data, self.logger)

            # Log collection results
            collection_method = collected_data.get('_metadata', {}).get('collection_method', 'unknown')
            self.logger.debug(
                f'Cluster info collected via {collection_method}: '
                f'{cluster_info.num_hosts} hosts, '
                f'{cluster_info.total_memory_bytes / (1024**3):.1f} GB total memory, '
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
        return generate_output_location(self, self.run_datetime)

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

        if not self.args.closed and not hasattr(self.args, "open"):
            self.logger.warning(f'Running the benchmark without verification for open or closed configurations. These results are not valid for submission. Use --open or --closed to specify a configuration.')
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
            if self.args.closed == False:
                # "--open" was passed
                self.logger.status(f'Running as allowed open configuration')
                return True
            else:
                self.logger.warning(f'Parameters allowed for open but not closed. Use --open and rerun the benchmark.')
                sys.exit(1)

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

    def run(self) -> int:
        """Execute the benchmark and track runtime.

        Wraps _run() with timing measurement. Updates self.runtime
        with the execution duration in seconds.

        Returns:
            Exit code from _run().
        """
        start_time = time.time()
        result = self._run()
        self.runtime = time.time() - start_time
        return result



