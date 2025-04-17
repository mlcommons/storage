#!/usr/bin/python3.9
#!/usr/bin/env python3

import abc
import concurrent.futures
import io
import os.path
import pprint
import select
import shlex
import signal
import subprocess
import sys
import time

from os.path import dirname
from typing import List, Dict, Union, Optional, Tuple

from mlpstorage.cli import parse_arguments, validate_args, update_args
from mlpstorage.config import *
from mlpstorage.logging import setup_logging, apply_logging_options
from mlpstorage.rules import calculate_training_data_size, generate_output_location, BenchmarkVerifier
from mlpstorage.utils import read_config_from_file, update_nested_dict, create_nested_dict, ClusterInformation, \
    CommandExecutor

logger = setup_logging("MLPerfStorage")

# Capturing TODO Items:
# Configure datasize to collect the memory information from the hosts instead of getting a number of hosts for the
#   calculation
#
# DONE Add logging module for better control of output messages
#
# Add function to generate DLIO command and manage execution
#
# Determine method to use cgroups for memory limitation in the benchmark script.
#
# Add a log block at the start of datagen & run that output all the parms being used to be clear on what a run is.

# Change num accelerators for datasize to "max num accelerators"
# for datagen change to num generation processes
# For run benchmark it stays the same


# Remove accelerator type from datagen
# datasize should output the datagen command to copy and paste

# Add autosize parameter for run_benchmark and datasize
# for run it's just size of dataset based on memory capacity
# For datasize it needs an input of GB/s for the cluster and list of hosts

# Keep a log of mlperfstorage commands executed in a mlperf.history file in results_dir

# Add support for datagen to use subdirectories
# Capture cluster information and write to a json document in outputdir. Figure out how to get all clients for milvus


def generate_mpi_prefix_cmd(mpi_cmd, hosts, num_processes, oversubscribe, allow_run_as_root):
    if mpi_cmd == MPIRUN:
        prefix = f"{MPI_RUN_BIN} -n {num_processes} -host {','.join(hosts)}"
    elif mpi_cmd == MPIEXEC:
        raise NotImplementedError(f"Unsupported MPI command: {mpi_cmd}")
    else:
        raise ValueError(f"Unsupported MPI command: {mpi_cmd}")

    if oversubscribe:
        prefix += " --oversubscribe"

    if allow_run_as_root:
        prefix += " --allow-run-as-root"

    return prefix


class Benchmark(abc.ABC):

    BENCHMARK_TYPE = None

    def __init__(self, args, run_number=0):
        self.args = args
        self.run_number = run_number
        validate_args(args)

        self.benchmark_verifier = BenchmarkVerifier(self, logger=logger)
        self.cmd_executor = CommandExecutor(logger=logger, debug=args.debug)

    def _execute_command(self, command, print_stdout=True, print_stderr=True) -> Tuple[str, str, int]:
        """
        Execute the given command and return stdout, stderr, and return code.
        :param command: Command to execute
        :param print_stdout: Whether to print stdout
        :param print_stderr: Whether to print stderr
        :return: (stdout, stderr, return code)
        """

        if self.args.what_if:
            logger.debug(f'Executing command in --what-if mode means no execution will be performed.')
            log_message = f'What-if mode: \nCommand: {command}'
            if self.args.debug:
                log_message += f'\n\nParameters: \n{pprint.pformat(vars(self.args))}'
            logger.info(log_message)
            return "", "", 0
        else:
            watch_signals = {signal.SIGINT, signal.SIGTERM}
            stdout, stderr, return_code = self.cmd_executor.execute(command, watch_signals=watch_signals,
                                                                    print_stdout=print_stdout, print_stderr=print_stderr)
            return stdout, stderr, return_code

    def _generate_output_location(self):
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to generate output location')
        return generate_output_location(self, DATETIME_STR)

    def verify_benchmark(self):
        if not self.BENCHMARK_TYPE:
            raise ValueError(f'No benchmark specified. Unable to verify benchmark')
        valid = self.benchmark_verifier.verify()
        if not valid:
            logger.error(f"Benchmark verification failed. Please check the logs for details.")
            sys.exit(1)

    @abc.abstractmethod
    def run(self):
        """
        Run the command for the given benchmark
        :return:
        """
        raise NotImplementedError


class TrainingBenchmark(Benchmark):

    TRAINING_CONFIG_PATH = "dlio/training"
    BENCHMARK_TYPE = BENCHMARK_TYPES.training

    def __init__(self, args):
        super().__init__(args)

        # This allows each command to map to a specific wrapper method. When methods are created, replace the default
        # 'self.execute_command' with the command-specific method (like "self._datasize()")
        self.command_method_map = dict(
            datasize=self.datasize,
            datagen=self._execute_command,
            run=self._execute_command,
            configview=self._execute_command,
            reportgen=self._execute_command,
        )

        self.per_host_mem_kB = None
        self.total_mem_kB = None

        self.params_dict = dict() if not args.params else {k: v for k, v in (item.split("=") for item in args.params)}

        self.base_command_path = f"dlio_benchmark"

        config_suffix = "datagen" if args.command == "datagen" else args.accelerator_type
        self.config_file = f"{args.model}_{config_suffix}.yaml"
        self.config_name = f"{args.model}_{config_suffix}"
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.TRAINING_CONFIG_PATH)

        self.run_result_output = self.generate_output_location()

        self.yaml_params = read_config_from_file(os.path.join(self.TRAINING_CONFIG_PATH, self.config_file))

        logger.info(f'nested: {create_nested_dict(self.params_dict)}')
        self.combined_params = update_nested_dict(self.yaml_params, create_nested_dict(self.params_dict))

        self.cluster_information = ClusterInformation(hosts=self.args.hosts, username=args.ssh_username,
                                                      debug=self.args.debug)

        self.verify_benchmark()

        logger.debug(f'yaml params: \n{pprint.pformat(self.yaml_params)}')
        logger.debug(f'combined params: \n{pprint.pformat(self.combined_params)}')
        logger.debug(f'Instance params: \n{pprint.pformat(self.__dict__)}')
        logger.status(f'Instantiated the Training Benchmark...')

    def generate_output_location(self):
        return self._generate_output_location()

    def run(self):
        self.command_method_map[self.args.command]()

    def generate_command(self):
        cmd = ""

        # Set the config file to use for params not passed via CLI
        if self.args.command in ["datagen", "run"]:
            cmd = f"{self.base_command_path}"
            cmd += f" --config-dir={self.config_path}"
            cmd += f" --config-name={self.config_name}"
        else:
            raise ValueError(f"Unsupported command: {self.args.command}")

        # Run directory for Hydra to output log files
        cmd += f" ++hydra.run.dir={self.run_result_output}"

        # Set the dataset directory and checkpoint directory
        if self.args.data_dir:
            cmd += f" ++workload.dataset.data_folder={self.args.data_dir}"
            cmd += f" ++workload.checkpoint.checkpoint_folder={self.args.data_dir}"

        # Configure the workflow depending on command
        if self.args.command == "datagen":
            cmd += " ++workload.workflow.generate_data=True ++workload.workflow.train=False"
        elif self.args.command == "run_benchmark":
            cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=True"

        # Training doesn't do checkpoints
        cmd += " ++workload.workflow.checkpoint=False"

        if self.params_dict:
            for key, value in self.params_dict.items():
                cmd += f" ++workload.{key}={value}"

        if self.args.exec_type == EXEC_TYPE.MPI:
            mpi_prefix = generate_mpi_prefix_cmd(MPIRUN, self.args.hosts, self.args.num_processes,
                                                 self.args.oversubscribe, self.args.allow_run_as_root)
            cmd = f"{mpi_prefix} {cmd}"

        return cmd

    def generate_datagen_benchmark_command(self, num_files_train, num_subfolders_train, num_samples_per_file):
        """
        This function will generate the command to use to call this program with the training & datagen parameters.
        """
        kv_map = {
            "num_files_train": num_files_train,
            "num_subfolders_train": num_subfolders_train,
            "num_samples_per_file": num_samples_per_file,
        }

        cmd = f"{sys.executable} {os.path.abspath(__file__)} training datagen"
        if self.args.hosts:
            cmd += f" --hosts={self.args.hosts}"
        cmd += f" --model={self.args.model}"
        cmd += f" --exec-type={self.args.exec_type}"
        if self.args.ssh_username:
            cmd += f" --ssh-username={self.args.ssh_username}"

        if self.params_dict:
            for key, value in self.params_dict.items():
                if key in kv_map.keys():
                    continue
                cmd += f" --{key}={value}"

        for key, value in kv_map.items():
            cmd += f" --param {key}={value}"

        # During datasize, this will be set to max_accelerators
        cmd += f" --num-processes={self.args.num_processes}"
        cmd += f" --results-dir={self.args.results_dir}"
        cmd += f" --data-dir=<INSERT_DATA_DIR>"

        return cmd

    def execute_command(self):
        cmd = self.generate_command()
        self._execute_command(cmd)

    def datasize(self):
        num_files_train, num_subfolders_train, num_samples_per_file = calculate_training_data_size(
            self.args, self.cluster_information, self.combined_params['dataset'], self.combined_params['reader'], logger
        )
        logger.result(f'Number of training files: {num_files_train}')
        logger.result(f'Number of training subfolders: {num_subfolders_train}')
        logger.result(f'Number of training samples per file: {num_samples_per_file}')

        cmd = self.generate_datagen_benchmark_command(num_files_train, num_subfolders_train, num_samples_per_file)
        logger.result(f'Run the following command to generate data: \n{cmd}')
        logger.warning(f'The parameter for --num-processes is the same as --max-accelerators. Adjust the value '
                       f'according to your system.')


class VectorDBBenchmark(Benchmark):

    VECTORDB_CONFIG_PATH = "vectordbbench"
    VDBBENCH_BIN = "vdbbench"
    BENCHMARK_TYPE = BENCHMARK_TYPES.vector_database

    def __init__(self, args):
        super().__init__(args)
        self.command_method_map = {
            "datagen": self.execute_datagen,
            "run-search": self.execute_run
        }

        self.command = args.command
        self.category = args.category if hasattr(args, 'category') else None
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.VECTORDB_CONFIG_PATH)
        self.config_name = args.config if hasattr(args, 'config') and args.config else "default"
        self.yaml_params = read_config_from_file(os.path.join(self.config_path, f"{self.config_name}.yaml"))
        self.run_result_output = self.generate_output_location()

        self.verify_benchmark()
        
        logger.status(f'Instantiated the VectorDB Benchmark...')
        
    def generate_output_location(self):
        """
        Generate the output directory structure for vector database benchmark results
        """
        return self._generate_output_location()
    
    def run(self):
        """Execute the appropriate command based on the command_method_map"""
        if self.command in self.command_method_map:
            logger.verboser(f"Executing command: {self.command}")
            self.command_method_map[self.command]()
        else:
            logger.error(f"Unsupported command: {self.command}")
            sys.exit(1)
    
    def build_command(self, script_name, additional_params=None):
        """
        Build a command string for executing a script with appropriate parameters
        
        Args:
            script_name (str): Name of the script to execute (e.g., "load_vdb.py" or "simple_bench.py")
            additional_params (dict, optional): Additional parameters to add to the command
            
        Returns:
            str: The complete command string
        """
        # Ensure output directory exists
        os.makedirs(self.run_result_output, exist_ok=True)
        
        # Build the base command
        config_file = os.path.join(self.config_path, f"{self.config_name}.yaml")
        
        cmd = f"{script_name}"
        cmd += f" --config {config_file}"
        cmd += f" --output-dir {self.run_result_output}"
        
        # Add host and port if provided (common to both datagen and run)
        if hasattr(self.args, 'host') and self.args.host:
            cmd += f" --host {self.args.host}"
        if hasattr(self.args, 'port') and self.args.port:
            cmd += f" --port {self.args.port}"
            
        # Add any additional parameters
        if additional_params:
            for param, attr in additional_params.items():
                if attr:
                    cmd += f" --{param} {attr}"
                    
        return cmd
    
    def execute_datagen(self):
        """Execute the data generation command using load_vdb.py"""
        cmd = self.build_command("load-vdb")
            
        logger.verbose(f'Executing data generation.')
        self._execute_command(cmd)
    
    def execute_run(self):
        """Execute the benchmark run command using simple_bench.py"""
        # Define additional parameters specific to the run command
        additional_params = {
            "processes": self.args.num_query_processes,
            "runtime": self.args.runtime,
            "queries": self.args.queries,
        }
        
        cmd = self.build_command("vdbbench", additional_params)
        logger.verbose(f'Execuging benchmark run.')
        self._execute_command(cmd)


# Main function to handle command-line arguments and invoke the corresponding function.
def main():
    args = parse_arguments()

    apply_logging_options(logger, args)

    validate_args(args)
    update_args(args)
    program_switch_dict = dict(
        training=TrainingBenchmark,
        vectordb=VectorDBBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    benchmark = benchmark_class(args)
    benchmark.run()


if __name__ == "__main__":
    main()
