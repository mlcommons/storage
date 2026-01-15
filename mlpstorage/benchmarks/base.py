import abc
import json
import os
import pprint
import signal
import sys
import time
import types

from typing import Tuple
from functools import wraps

from pyarrow.ipc import open_stream

from mlpstorage.config import PARAM_VALIDATION, DATETIME_STR, MLPS_DEBUG, EXEC_TYPE
from mlpstorage.debug import debug_tryer_wrapper
from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.rules import BenchmarkVerifier, generate_output_location, ClusterInformation
from mlpstorage.utils import CommandExecutor, MLPSJsonEncoder
from mlpstorage.cluster_collector import collect_cluster_info


class Benchmark(abc.ABC):

    BENCHMARK_TYPE = None

    def __init__(self, args, logger=None, run_datetime=None, run_number=0) -> None:
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

        self.benchmark_run_verifier = None
        self.verification = None
        self.cmd_executor = CommandExecutor(logger=self.logger, debug=args.debug)

        self.command_output_files = list()
        self.run_result_output = self.generate_output_location()
        os.makedirs(self.run_result_output, exist_ok=True)

        self.metadata_filename = f"{self.BENCHMARK_TYPE.value}_{self.run_datetime}_metadata.json"
        self.metadata_file_path = os.path.join(self.run_result_output, self.metadata_filename)

        self.logger.status(f'Benchmark results directory: {self.run_result_output}')

    def _execute_command(self, command, output_file_prefix=None, print_stdout=True, print_stderr=True) -> Tuple[str, str, int]:
        """
        Execute the given command and return stdout, stderr, and return code.
        :param command: Command to execute
        :param print_stdout: Whether to print stdout
        :param print_stderr: Whether to print stderr
        :return: (stdout, stderr, return code)
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
    def metadata(self):
        """
        Generate metadata dict capturing the benchmark run configuration.

        This metadata is designed to be complete enough that BenchmarkRunData
        can be reconstructed from it without needing tool-specific result files.
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

    def write_metadata(self):
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
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to generate output location')
        return generate_output_location(self, self.run_datetime)

    def verify_benchmark(self) -> bool:
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
    def _run(self):
        """
        Run the command for the given benchmark.
        :return:
        """
        raise NotImplementedError

    def run(self):
        start_time = time.time()
        result = self._run()
        self.runtime = time.time() - start_time
        return result



