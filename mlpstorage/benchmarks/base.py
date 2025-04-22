import abc
import pprint
import signal
import sys

from typing import Tuple
from functools import wraps

from mlpstorage.config import PARAM_VALIDATION, DATETIME_STR
from mlpstorage.logging import setup_logging, apply_logging_options
from mlpstorage.rules import BenchmarkVerifier, generate_output_location
from mlpstorage.utils import CommandExecutor


class Benchmark(abc.ABC):

    BENCHMARK_TYPE = None

    def __init__(self, args, logger=None, run_datetime=None, run_number=0) -> None:
        self.args = args
        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name=f"{self.BENCHMARK_TYPE}_benchmark", stream_log_level=args.stream_log_level)
            apply_logging_options(self.logger, args)

        self.run_datetime = run_datetime if run_datetime else DATETIME_STR
        self.run_number = run_number

        self.benchmark_verifier = BenchmarkVerifier(self, logger=self.logger)
        self.cmd_executor = CommandExecutor(logger=self.logger, debug=args.debug)

    def __getattribute__(self, name):
        """
        Special method to intercept attribute access.
        If the attribute is the 'run' method, wrap it with timing functionality.
        """
        attr = super().__getattribute__(name)

        # Only intercept the 'run' method
        if name == 'run' and callable(attr):
            @wraps(attr)
            def timed_run(*args, **kwargs):
                # Get logger if available
                try:
                    logger = super(Benchmark, self).__getattribute__('logger')
                    logger.info(f"Starting benchmark run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    start_time = time.time()

                    # Execute the original run method
                    result = attr(*args, **kwargs)

                    # Calculate and log the execution time
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # Format the time nicely
                    if execution_time < 60:
                        time_str = f"{execution_time:.2f} seconds"
                    elif execution_time < 3600:
                        minutes = int(execution_time // 60)
                        seconds = execution_time % 60
                        time_str = f"{minutes} minutes and {seconds:.2f} seconds"
                    else:
                        hours = int(execution_time // 3600)
                        minutes = int((execution_time % 3600) // 60)
                        seconds = execution_time % 60
                        time_str = f"{hours} hours, {minutes} minutes and {seconds:.2f} seconds"

                    logger.info(f"Benchmark run completed in {time_str}")
                    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Store the execution time as an attribute for potential later use
                    super(Benchmark, self).__setattr__('last_run_duration', execution_time)

                    return result
                except (AttributeError, Exception) as e:
                    # If logger is not available or any other error occurs,
                    # just run the original method without timing
                    return attr(*args, **kwargs)

            return timed_run

        # Return the original attribute for all other cases
        return attr

    def _execute_command(self, command, print_stdout=True, print_stderr=True) -> Tuple[str, str, int]:
        """
        Execute the given command and return stdout, stderr, and return code.
        :param command: Command to execute
        :param print_stdout: Whether to print stdout
        :param print_stderr: Whether to print stderr
        :return: (stdout, stderr, return code)
        """

        if self.args.what_if:
            self.logger.debug(f'Executing command in --what-if mode means no execution will be performed.')
            log_message = f'What-if mode: \nCommand: {command}'
            if self.args.debug:
                log_message += f'\n\nParameters: \n{pprint.pformat(vars(self.args))}'
            self.logger.info(log_message)
            return "", "", 0
        else:
            watch_signals = {signal.SIGINT, signal.SIGTERM}
            stdout, stderr, return_code = self.cmd_executor.execute(command, watch_signals=watch_signals,
                                                                    print_stdout=print_stdout, print_stderr=print_stderr)
            return stdout, stderr, return_code

    def _generate_output_location(self) -> str:
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to generate output location')
        return generate_output_location(self, self.run_datetime)

    def verify_benchmark(self) -> bool:
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to verify benchmark')
        validation = self.benchmark_verifier.verify()
        if validation == PARAM_VALIDATION.CLOSED:
            return True
        if validation == PARAM_VALIDATION.INVALID:
            if self.args.allow_invalid_config:
                self.logger.warning(f'Invalid configuration found. Allowing the benchmark to proceed.')
                return True
            sys.exit(1)
        if validation == PARAM_VALIDATION.OPEN:
            if self.args.closed == False:
                # "--open" was passed
                self.logger.status(f'Running as allowed open configuration')
            else:
                self.logger.warning(f'Parameters allowed for open but not closed. Use --open and rerun the benchmark.')
                sys.exit(1)

    @abc.abstractmethod
    def run(self):
        """
        Run the command for the given benchmark
        :return:
        """
        raise NotImplementedError
