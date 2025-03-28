#!/usr/bin/env python3

import abc
import concurrent.futures
import os.path
import pprint
import subprocess
import sys

from benchmark.cli import parse_arguments, validate_args
from benchmark.config import *
from benchmark.logging import setup_logging
from benchmark.rules import validate_dlio_parameter
from benchmark.utils import read_config_from_file, update_nested_dict, create_nested_dict, ClusterInformation


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

    def __init__(self, args, run_number=None):
        self.args = args
        self.run_number = run_number
        validate_args(args)

    @abc.abstractmethod
    def run(self):
        """
        Run the command for the given benchmark
        :return:
        """
        raise NotImplementedError


class TrainingBenchmark(Benchmark):

    TRAINING_CONFIG_PATH = "configs/dlio/training"

    def __init__(self, args):
        super().__init__(args)

        # This allows each command to map to a specific wrapper method. When meethods are created, repalce the default
        # 'self.execute_command' with the command-specific method (like "self._datasize()")
        self.command_method_map = dict(
            datasize=self._datasize,
            datagen=self.execute_command,
            run_benchmark=self.execute_command,
            configview=self.execute_command,
            reportgen=self.execute_command,
        )

        self.per_host_mem_kB = None
        self.total_mem_kB = None

        self.params_dict = dict() if not args.params else {k: v for k, v in (item.split("=") for item in args.params)}

        self.base_command_path = f"{sys.executable} dlio_benchmark/dlio_benchmark/main.py"

        config_suffix = "datagen" if args.command == "datagen" else args.accelerator_type
        self.config_path = f"{args.model}_{config_suffix}.yaml"
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAINING_CONFIG_PATH)
        self.config_name = f"{args.model}_{config_suffix}"

        self.run_result_output = self.generate_output_location()

        self.yaml_params = read_config_from_file(os.path.join(self.config_path, f"{self.config_name}.yaml"))
        self.validate_params()

        logger.info(f'nested: {create_nested_dict(self.params_dict)}')
        self.combined_params = update_nested_dict(self.yaml_params, create_nested_dict(self.params_dict))

        self.cluster_information = ClusterInformation(hosts=self.args.hosts, debug=self.args.debug)

        logger.debug(f'yaml params: \n{pprint.pformat(self.yaml_params)}')
        logger.debug(f'combined params: \n{pprint.pformat(self.combined_params)}')
        logger.debug(f'Instance params: \n{pprint.pformat(self.__dict__)}')
        logger.status(f'Instantiated the Training Benchmark...')

    def generate_output_location(self):
        """
        Output structure:
        RESULTS_DIR:
          unet3d:
            training:
              DATETIME:
                run_1
                ...
                run_5
            datagen:
              DATETIME:
                log_files
          llama3
            checkpoint
              DATETIME:
                run_1
                ...
                run_10
            recovery
              DATETIME:
                run_1
                ...
                run_5
          vectordb:
            throughput:

        If benchmark.py is not doing multiple runs then the results will be in a directory run_0
        :return:
        """
        output_location = self.args.results_dir
        output_location = os.path.join(output_location, self.args.model)
        output_location = os.path.join(output_location, self.args.command)
        output_location = os.path.join(output_location, DATETIME_STR)

        if self.args.command == "run":
            output_location = os.path.join(output_location, f"run_{self.run_number}")

        return output_location

    def run(self):
        self.command_method_map[self.args.command]()

    def validate_params(self):
        # Add code here for validation processes. We do not need to validate an option is in a list as the argparse
        #  option "choices" accomplishes this for us.
        validation_results = dict()
        any_non_closed = False
        if self.params_dict:
            for param, value in self.params_dict.items():
                validation_results[param] = [self.args.model, value, validate_dlio_parameter(self.args.model, param, value)]
                if validation_results[param][2] != PARAM_VALIDATION.CLOSED:
                    any_non_closed = True

        if any_non_closed:
            error_string = "\n\t".join([f"{p} = {v[1]}" for p, v in validation_results.items()])
            logger.error(f'\nNot all parameters allowed in closed submission: \n'
                                  f'\t{error_string}')
            if not self.args.allow_invalid_params:
                print("Invalid parameters found. Please check the command and parameters.")
                sys.exit(1)

    def generate_command(self):
        cmd = ""

        # Set the config file to use for params not passed via CLI
        if self.args.command in ["datagen", "run_benchmark"]:
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
                cmd += f" ++{key}={value}"

        if self.args.exec_type == EXEC_TYPE.MPI:
            mpi_prefix = generate_mpi_prefix_cmd(MPIRUN, self.args.hosts, self.args.num_processes,
                                                 self.args.oversubscribe, self.args.allow_run_as_root)
            cmd = f"{mpi_prefix} {cmd}"

        return cmd

    def execute_command(self):
        cmd = self.generate_command()

        if self.args.debug:
            logger.status(f'Executing: {cmd}')
            return

        logger.info(f'Executing: {cmd}')
        subprocess.call(cmd, shell=True)

    def _datasize(self):
        """
        Validate the parameters for the datasize operation and apply rules for a closed submission.

        Requirements:
          - Dataset needs to be 5x the amount of total memory
          - Training needs to do at least 500 steps per epoch

        Memory Ratio:
          - Collect "Total Memory" from /proc/meminfo on each host
          - sum it up
          - multiply by 5
          - divide by sample size
          - divide by batch size

        500 steps:
          - 500 steps per ecpoch
          - multiply by max number of processes
          - multiply by batch size
        :return:
        """

        logger.info(f'Got to datasize')


class VectorDBBenchmark(Benchmark):
    VECTORDB_CONFIG_PATH = "configs/vector_db"

    def __init__(self, command, category=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.command_method_map = dict(
            throughput=self._throughput,
            latency=self._latency
        )
        self.command = command
        self.category = category

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.VECTORDB_CONFIG_PATH)

    def run(self):
        self.command_method_map[self.command]()

    def _throughput(self):
        logger.info(f'Got to throughput')

    def _latency(self):
        logger.info(f'Got to latency')


# Main function to handle command-line arguments and invoke the corresponding function.
def main(args):
    validate_args(args)
    program_switch_dict = dict(
        training=TrainingBenchmark,
        vectordb=VectorDBBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    benchmark = benchmark_class(args)
    benchmark.run()


if __name__ == "__main__":
    # Get the mllogger and args. Call main to run program
    cli_args = parse_arguments()
    logger = setup_logging("MLPerfStorage")
    main(cli_args)
