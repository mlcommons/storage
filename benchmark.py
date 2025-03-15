#!/usr/bin/env python3

import argparse
import concurrent.futures
import datetime
import enum
import os
import os.path
import pprint
import subprocess
import sys
import yaml

from mlperf_logging import mllog

import logging

# Define the custom log levels
STATUS = 25
VERBOSE = 13
VERBOSER = 12
VERBOSEST = 11

STREAM_LOG_LEVEL = logging.DEBUG

COLOR_MAP = {
    'normal': "\033[0m",
    'white': "\033[37m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    'intense_purple': '\033[1;95m',
}

# Create a custom logger
logger = logging.getLogger('custom_logger')
logger.setLevel(STREAM_LOG_LEVEL)

# Define the custom log levels
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(VERBOSER, "VERBOSER")
logging.addLevelName(VERBOSEST, "VERBOSEST")
logging.addLevelName(STATUS, "STATUS")


# Define the custom log level methods
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


def verboser(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSER):
        self._log(VERBOSER, message, args, **kwargs)


def verbosest(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSEST):
        self._log(VERBOSEST, message, args, **kwargs)


def status(self, message, *args, **kwargs):
    if self.isEnabledFor(STATUS):
        self._log(STATUS, message, args, **kwargs)


# Create a custom formatter for the logger
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color_map = {
            logging.DEBUG: COLOR_MAP['white'],
            logging.WARNING: COLOR_MAP['yellow'],
            logging.ERROR: COLOR_MAP['red'],
            logging.CRITICAL: COLOR_MAP['red'],
            STATUS: COLOR_MAP['intense_purple'],
        }
        formatted_time = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        color = color_map.get(record.levelno, COLOR_MAP['normal'])
        return f"{color}{formatted_time}|{record.levelname}:{record.module}:{record.lineno}: " \
               f"{record.getMessage()}{COLOR_MAP['normal']}"


# Add the custom log level methods to the logger
logging.Logger.verbose = verbose
logging.Logger.verboser = verboser
logging.Logger.verbosest = verbosest
logging.Logger.status = status

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColoredFormatter())
stream_handler.setLevel(STREAM_LOG_LEVEL)
logger.addHandler(stream_handler)


# Define constants:
COSMOFLOW = "cosmoflow"
RESNET = "resnet50"
UNET = "unet3d"
MODELS = [COSMOFLOW, RESNET, UNET]

H100 = "h100"
A100 = "a100"
ACCELERATORS = [H100, A100]

OPEN = "open"
CLOSED = "closed"
CATEGORIES = [OPEN, CLOSED]

LLAMA3_70B = 'llama3-70b'
LLAMA3_405B = 'llama3-405b'
LLM_1620B = 'llm-1620b'
LLM_MODELS = [LLAMA3_70B, LLAMA3_405B, LLM_1620B]

LLM_MODEL_PARALLELISMS = {
    LLAMA3_70B: dict(tp=8, pp=4, min_ranks=4 * 8),
    LLAMA3_405B: dict(tp=8, pp=16, min_ranks=8 * 16),
    LLM_1620B: dict(tp=8, pp=64, min_ranks=64 * 8),
}
MIN_RANKS_STR = "; ".join(
    [f'{key} = {value["min_ranks"]} accelerators' for key, value in LLM_MODEL_PARALLELISMS.items()])

MPIRUN = "mpirun"
MPIEXEC = "mpiexec"
MPI_CMDS = [MPIRUN, MPIEXEC]

STEPS_PER_EPOCH = 500
MOST_MEMORY_MULTIPLIER = 5
MAX_READ_THREADS_TRAINING = 32

DEFAULT_HOSTS = "127.0.0.1"

MPI_RUN_BIN = os.environ.get("MPI_RUN_BIN", MPIRUN)
ALLOW_RUN_AS_ROOT = True

class EXEC_TYPE(enum.Enum):
    MPI = "mpi"
    DOCKER = "docker"


class PARAM_VALIDATION(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    INVALID = "invalid"


DATETIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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


def parse_arguments():
    # Many of the help messages are shared between the subparsers. This dictionary prevents rewriting the same messages
    # in multiple places.
    help_messages = dict(
        model="Model to emulate. A specific model defines the sample size, sample container format, and data "
              "rates for each supported accelerator.",
        accelerator_type="Accelerator to simulate for the benchmark. A specific accelerator defines the data access "
                         "sizes and rates for each supported workload",
        num_accelerators_datasize="Simulated number of accelerators. In multi-host configurations the accelerators "
                                  "will be initiated in a round-robin fashion to ensure equal distribution of "
                                  "simulated accelerator processes",
        num_accelerators_datagen="Number of parallel processes to use for dataset generation. Processes will be "
                                 "initiated in a round-robin fashion across the configured client hosts",
        num_client_hosts="Number of participating client hosts. Simulated accelerators will be initiated on these "
                         "hosts in a round-robin fashion",
        client_host_mem_GB="Memory available in the client where the benchmark is run. The dataset needs to be 5x the "
                           "available memory for closed submissions.",
        client_hosts="Comma separated IP addresses of the participating hosts (without spaces). "
                     "eg: '192.168.1.1,192.168.1.2'",
        category="Benchmark category to be submitted.",
        results_dir="Directory where the benchmark results will be saved.",
        params="Additional parameters to be passed to the benchmark. These will override the config file. For a closed "
               "submission only a subset of params are supported. Multiple values allowed in the form: "
               "--params key1=value1 key2=value2 key3=value3",
        datasize="The datasize command calculates the number of samples needed for a given workload, accelerator type,"
                 " number of accelerators, and client host memory.",
        datagen="The datagen command generates a dataset for a given workload and number of parallel generation "
                "processes.",
        run_benchmark="Run the benchmark with the specified parameters.",
        configview="View the final config based on the specified options.",
        reportgen="Generate a report from the benchmark results.",

        # Checkpointing help messages
        checkpoint="The checkpoint command executes checkpoints in isolation as a write-only workload",
        recovery="The recovery command executes a recovery of the most recently written checkpoint with randomly "
                 "assigned reader to data mappings",
        llm_model="The model & size to be emulated for checkpointing. The selection will dictate the TP, PP, & DP "
                  "sizes as well as the size of the checkpoint",
        num_checkpoint_accelerators=f"The number of accelerators to emulate for the checkpoint task. Each LLM Model "
                                    f"can be executed as 8 accelerators or the minimum required to run the model: "
                                    f"\n{MIN_RANKS_STR}"
    )

    parser = argparse.ArgumentParser(description="Script to launch the MLPerf Storage benchmark")
    parser.add_argument("--allow-invalid-params", "-aip", action="store_true", help="Do not fail on invalid parameters.")
    sub_programs = parser.add_subparsers(dest="program", required=True, help="Sub-programs")
    sub_programs.required = True

    # Training
    training_parsers = sub_programs.add_parser("training", help="Training benchmark options")
    training_parsers.add_argument("--data-dir", '-dd', type=str, help="Filesystem location for data")
    training_parsers.add_argument('--results-dir', '-rd', type=str, required=True, help=help_messages['results_dir'])
    training_subparsers = training_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    training_parsers.required = True

    datasize = training_subparsers.add_parser("datasize", help=help_messages['datasize'])
    datasize.add_argument('--model', '-m', choices=MODELS, required=True, help=help_messages['model'])
    datasize.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, required=True, help=help_messages['accelerator_type'])
    datasize.add_argument('--num-accelerators', '-na', type=int, required=True, help=help_messages['num_accelerators_datasize'])
    datasize.add_argument('--num-client-hosts', '-nc', type=int, required=True, help=help_messages['num_client_hosts'])
    datasize.add_argument('--client-host-memory-in-gb', '-cm', type=int, required=True, help=help_messages['client_host_mem_GB'])

    datagen = training_subparsers.add_parser("datagen", help=help_messages['datagen'])
    datagen.add_argument('--hosts', '-s', type=str, default=DEFAULT_HOSTS, help=help_messages['client_hosts'])
    datagen.add_argument('--category', '-c', choices=CATEGORIES, help=help_messages['category'])
    datagen.add_argument('--model', '-m', choices=MODELS, required=True, help=help_messages['model'])
    datagen.add_argument('--accelerator-type', '-a', choices=ACCELERATORS, required=True, help=help_messages['accelerator_type'])
    datagen.add_argument('--num-accelerators', '-n', type=int, required=True, help=help_messages['num_accelerators_datagen'])
    datagen.add_argument('--params', '-p', nargs="+", type=str, help=help_messages['params'])

    run_benchmark = training_subparsers.add_parser("run", help=help_messages['run_benchmark'])
    run_benchmark.add_argument('--hosts', '-s', type=str, default=DEFAULT_HOSTS, help=help_messages['client_hosts'])
    run_benchmark.add_argument('--category', '-c', choices=CATEGORIES, help=help_messages['category'])
    run_benchmark.add_argument('--model', '-m', choices=MODELS, required=True, help=help_messages['model'])
    run_benchmark.add_argument('--accelerator-type', '-a', choices=ACCELERATORS, required=True, help=help_messages['accelerator_type'])
    run_benchmark.add_argument('--num-accelerators', '-n', type=int, required=True, help=help_messages['num_accelerators_datasize'])
    run_benchmark.add_argument('--params', '-p', nargs="+", type=str, help=help_messages['params'])

    configview = training_subparsers.add_parser("configview", help=help_messages['configview'])
    configview.add_argument('--model', '-m', choices=MODELS, help=help_messages['model'])
    configview.add_argument('--accelerator-type', '-a', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    configview.add_argument('--params', '-p', nargs="+", type=str, help=help_messages['params'])

    reportgen = training_subparsers.add_parser("reportgen", help=help_messages['reportgen'])
    reportgen.add_argument('--results-dir', '-r', type=str, help=help_messages['results_dir'])

    mpi_options = training_parsers.add_argument_group("MPI")
    mpi_options.add_argument('--oversubscribe', action="store_true")
    # mpi_options.add_argument('--allow-run-as-root', action="store_true")


    # Checkpointing
    checkpointing_parsers = sub_programs.add_parser("checkpointing", help="Checkpointing benchmark options")
    checkpointing_subparsers = checkpointing_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    checkpointing_parsers.required = True

    # Add specific checkpointing benchmark options here
    checkpoint = checkpointing_subparsers.add_parser('checkpoint', help=help_messages['checkpoint'])
    checkpoint.add_argument('--llm-model', '-lm', choices=LLM_MODELS, help=help_messages['llm_model'])
    checkpoint.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])
    checkpoint.add_argument('--num-accelerators', '-na', type=int, help=help_messages['num_checkpoint_accelerators'])

    recovery = checkpointing_subparsers.add_parser('recovery', help=help_messages['recovery'])

    # VectorDB Benchmark
    vectordb_parsers = sub_programs.add_parser("vectordb", help="VectorDB benchmark options")
    vectordb_parsers.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])

    vectordb_subparsers = vectordb_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    vectordb_parsers.required = True

    # Add specific VectorDB benchmark options here
    throughput = vectordb_subparsers.add_parser('throughput', help=help_messages['vectordb_throughput'])
    latency = vectordb_subparsers.add_parser('latency', help=help_messages['vectordb_latency'])

    return parser.parse_args()


def validate_args(args):
    error_messages = []
    # Add generic validations here. Workload specific validation is in the Benchmark classes

    if error_messages:
        for msg in error_messages:
            print(msg)

        sys.exit(1)


def read_config_from_file(relative_path):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


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


def validate_dlio_parameter(model, param, value):
    """
    This function applies rules to the allowed changeable dlio parameters.
    """
    if model in MODELS:
        # Allowed to change data_folder and number of files to train depending on memory requirements
        if param.startswith('dataset'):
            left, right = param.split('.')
            if right in ('data_folder', 'num_files_train'):
                # TODO: Add check of min num_files for given memory config
                return PARAM_VALIDATION.CLOSED

        # Allowed to set number of read threads
        if param.startswith('reader'):
            left, right = param.split('.')
            if right == "read_threads":
                if 0 < int(value) < MAX_READ_THREADS_TRAINING:
                    return PARAM_VALIDATION.CLOSED

    elif model in LLM_MODELS:
        # TODO: Define params that can be modified in closed
        pass

    return PARAM_VALIDATION.INVALID


def update_nested_dict(original_dict, update_dict):
    updated_dict = {}
    for key, value in original_dict.items():
        if key in update_dict:
            if isinstance(value, dict) and isinstance(update_dict[key], dict):
                updated_dict[key] = update_nested_dict(value, update_dict[key])
            else:
                updated_dict[key] = update_dict[key]
        else:
            updated_dict[key] = value
    for key, value in update_dict.items():
        if key not in original_dict:
            updated_dict[key] = value
    return updated_dict


def create_nested_dict(flat_dict, parent_dict=None, separator='.'):
    if parent_dict is None:
        parent_dict = {}

    for key, value in flat_dict.items():
        keys = key.split(separator)
        current_dict = parent_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value

    return parent_dict


class ClusterInformation:
    def __init__(self, hosts):
        self.hosts = hosts
        self.info = self.collect_info()

    def collect_info(self):
        info = {}
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
                 *args, **kwargs):

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
        cluster_info = ClusterInformation(hosts=self.hosts)
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
    logger.setLevel(logging.DEBUG)
    main(cli_args)
