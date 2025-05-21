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

from mlpstorage.config import PARAM_VALIDATION, DATETIME_STR, MLPS_DEBUG
from mlpstorage.debug import debug_tryer_wrapper
from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.rules import BenchmarkRunVerifier, generate_output_location
from mlpstorage.utils import CommandExecutor, MLPSJsonEncoder


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
        metadata = dict()
        keys_to_skip = ["command_method_map", "logger", 'benchmark_run_verifier', "cmd_executor"]
        for k, v in self.__dict__.items():
            if not k in keys_to_skip and not k.startswith("__"):
                metadata[k] = v

        metadata['benchmark_type'] = self.BENCHMARK_TYPE.name

        return metadata

    def write_metadata(self):
        with open(self.metadata_file_path, 'w+') as fd:
            json.dump(self.metadata, fd, indent=2, cls=MLPSJsonEncoder)

        if self.args.verbose or self.args.debug or self.debug:
            json.dump(self.metadata, sys.stdout, indent=2, cls=MLPSJsonEncoder)

    def generate_output_location(self) -> str:
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to generate output location')
        return generate_output_location(self, self.run_datetime)

    def verify_benchmark(self) -> bool:
        self.logger.verboser(f'Verifying benchmark parameters: {self.args}')
        if not self.benchmark_run_verifier:
            self.benchmark_run_verifier = BenchmarkRunVerifier(self, logger=self.logger)

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



