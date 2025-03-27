#!/usr/bin/env python3

import concurrent.futures
import os.path
import pprint
import subprocess
import sys

from benchmark.cli import parse_arguments, validate_args
from benchmark.config import *
from benchmark.logging import setup_logging
from benchmark.rules import validate_dlio_parameter
from benchmark.utils import read_config_from_file, update_nested_dict, create_nested_dict


# Capturing TODO Items:
# Configure datasize to collect the memory information from the hosts instead of getting a number of hosts for the
#   calculation
#
# Add logging module for better control of output messages
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
        prefix = f"{MPI_RUN_BIN} -n {num_processes} -host {hosts}"
    elif mpi_cmd == MPIEXEC:
        raise NotImplementedError(f"Unsupported MPI command: {mpi_cmd}")
    else:
        raise ValueError(f"Unsupported MPI command: {mpi_cmd}")

    if oversubscribe:
        prefix += " --oversubscribe"

    if allow_run_as_root:
        prefix += " --allow-run-as-root"

    return prefix



class ClusterInformation:
    def __init__(self, hosts, debug=False):
        self.debug = debug
        self.hosts = hosts
        self.info = self.collect_info()

    def collect_info(self):
        info = {}

        if self.debug:
            print(f"Collecting information for hosts: {self.hosts}")
            return {host: {'cpu_core_count': 0,'memory_info': {}} for host in self.hosts}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_host = {executor.submit(self.get_host_info, host): host for host in self.hosts}
            for future in concurrent.futures.as_completed(future_to_host):
                host = future_to_host[future]
                cpu_core_count, memory_info = future.result()
                info[host] = {
                    'cpu_core_count': cpu_core_count,
                    'memory_info': memory_info
                }
        return info

    def get_host_info(self, host):
        cpu_core_count = self.get_cpu_core_count(host)
        memory_info = self.get_memory_info(host)
        return cpu_core_count, memory_info

    def get_cpu_core_count(self, host):
        cpu_core_count = 0
        cpu_info_path = f"ssh {host} cat /proc/cpuinfo"
        try:
            output = os.popen(cpu_info_path).read()
            cpu_core_count = output.count('processor')
        except Exception as e:
            print(f"Error getting CPU core count for host {host}: {e}")
        return cpu_core_count

    def get_memory_info(self, host):
        memory_info = {}
        meminfo_path = f"ssh {host} cat /proc/meminfo"
        try:
            output = os.popen(meminfo_path).read()
            lines = output.split('\n')
            for line in lines:
                if line.startswith('MemTotal:'):
                    memory_info['total'] = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    memory_info['free'] = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    memory_info['available'] = int(line.split()[1])
        except Exception as e:
            print(f"Error getting memory information for host {host}: {e}")
        return memory_info


class Benchmark:
    def run(self):
        """
        Run the command for the given benchmark
        :return:
        """
        raise NotImplementedError


class TrainingBenchmark(Benchmark):

    TRAINING_CONFIG_PATH = "configs/dlio/training"

    def __init__(self, command, category=None, model=None, hosts=None, accelerator_type=None, num_accelerators=None,
                 client_host_memory_in_gb=None, num_client_hosts=None, params=None, oversubscribe=False,
                 allow_run_as_root=True, data_dir=None, results_dir=None, run_number=0, allow_invalid_params=False,
                 debug=False, *args, **kwargs):

        self.debug = debug

        # This allows each command to map to a specific wrapper method. When meethods are created, repalce the default
        # 'self.execute_command' with the command-specific method (like "self._datasize()")
        self.command_method_map = dict(
            datasize=self._datasize,
            datagen=self.execute_command,
            run_benchmark=self.execute_command,
            configview=self.execute_command,
            reportgen=self.execute_command,
        )

        self.command = command
        self.category = category
        self.model = model
        self.hosts = hosts.split(',')
        self.accelerator_type = accelerator_type
        self.num_accelerators = num_accelerators
        self.client_host_memory_in_gb = client_host_memory_in_gb
        self.num_client_hosts = num_client_hosts
        self.params_dict = dict() if not params else {k: v for k, v in (item.split("=") for item in params)}
        self.oversubscribe = oversubscribe
        self.allow_run_as_root = allow_run_as_root
        self.allow_invalid_params = allow_invalid_params

        self.results_dir = results_dir
        self.data_dir = data_dir
        self.run_number = run_number

        self.base_command_path = f"{sys.executable} dlio_benchmark/dlio_benchmark/main.py"
        self.exec_type = None  #EXEC_TYPE.MPI

        self.config_path = f"{self.model}_{self.accelerator_type}.yaml"
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAINING_CONFIG_PATH)
        self.config_name = f"{self.model}_{self.accelerator_type}"

        self.run_result_output = self.generate_output_location()

        self.yaml_params = read_config_from_file(os.path.join(self.config_path, f"{self.config_name}.yaml"))
        self.validate_params()
        logger.info(f'nested: {create_nested_dict(self.params_dict)}')
        self.combined_params = update_nested_dict(self.yaml_params, create_nested_dict(self.params_dict))

        self.per_host_mem_kB = None
        self.total_mem_kB = None
        self.cluster_information = self.get_cluster_information()

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
        output_location = self.results_dir
        output_location = os.path.join(output_location, self.model)
        output_location = os.path.join(output_location, self.command)
        output_location = os.path.join(output_location, DATETIME_STR)

        if self.command == "run":
            output_location = os.path.join(output_location, f"run_{self.run_number}")

        return output_location

    def run(self):
        self.command_method_map[self.command]()

    def validate_params(self):
        # Add code here for validation processes. We do not need to validate an option is in a list as the argparse
        #  option "choices" accomplishes this for us.
        validation_results = dict()
        any_non_closed = False
        if self.params_dict:
            for param, value in self.params_dict.items():
                validation_results[param] = [self.model, value, validate_dlio_parameter(self.model, param, value)]
                if validation_results[param][2] != PARAM_VALIDATION.CLOSED:
                    any_non_closed = True

        if any_non_closed:
            error_string = "\n\t".join([f"{p} = {v[1]}" for p, v in validation_results.items()])
            logger.error(f'\nNot all parameters allowed in closed submission: \n'
                                  f'\t{error_string}')
            if not self.allow_invalid_params:
                print("Invalid parameters found. Please check the command and parameters.")
                sys.exit(1)

    def get_cluster_information(self):
        cluster_info = ClusterInformation(hosts=self.hosts, debug=self.debug)
        logger.verbose(f'Cluster information: \n{pprint.pformat(cluster_info.info)}')
        # per_host_kb =
        return cluster_info.info

    def generate_command(self):
        cmd = ""

        if self.command in ["datagen", "run_benchmark"]:
            cmd = f"{self.base_command_path}"
            cmd += f" --config-dir={self.config_path}"
            cmd += f" --config-name={self.config_name}"
            # cmd += f" --workload={self.model}"
        else:
            raise ValueError(f"Unsupported command: {self.command}")

        cmd += f" ++hydra.run.dir={self.run_result_output}"

        if self.data_dir:
            cmd += f" ++workload.dataset.data_folder={self.data_dir}"
            cmd += f" ++workload.checkpoint.checkpoint_folder={self.data_dir}"

        if self.command == "datagen":
            cmd += " ++workload.workflow.generate_data=True ++workload.workflow.train=False"
        elif self.command == "run_benchmark":
            cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=True"

        cmd += " ++workload.workflow.checkpoint=False"

        if self.params_dict:
            for key, value in self.params_dict.items():
                cmd += f" ++{key}={value}"

        if self.exec_type == EXEC_TYPE.MPI:
            mpi_prefix = generate_mpi_prefix_cmd(MPIRUN, self.hosts, self.num_accelerators, self.oversubscribe, self.allow_run_as_root)
            cmd = f"{mpi_prefix} {cmd}"

        return cmd

    def execute_command(self):
        cmd = self.generate_command()
        logger.info(f'Executing: {cmd}')

        if self.debug:
            return

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
    benchmark = benchmark_class(**args.__dict__)
    benchmark.run()


if __name__ == "__main__":
    # Get the mllogger and args. Call main to run program
    cli_args = parse_arguments()
    logger = setup_logging("MLPerfStorage")
    main(cli_args)
