import os

from datetime import datetime
from typing import List, Tuple

from mlpstorage.config import (MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, BENCHMARK_TYPES,
                               DATETIME_STR, LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS)
from mlpstorage.logging import setup_logging
from mlpstorage.utils import is_valid_datetime_format


class BenchmarkVerifier:

    def __init__(self, benchmark, logger):
        self.benchmark = benchmark
        self.logger = logger

    def verify(self):
        # Training Verification
        if self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
            validation = self._verify_training_params()
        elif self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
            validation = self._verify_checkpointing_params()
        elif self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
            validation = self._verify_vector_database_params()
        else:
            validation = PARAM_VALIDATION.INVALID

        self.logger.status(f'Benchmark verification: {validation.name}')
        return validation

    def _verify_training_params(self) -> PARAM_VALIDATION:
        # Add code here for validation processes. We do not need to validate an option is in a list as the argparse
        #  option "choices" accomplishes this for us.

        # We will walk through all the params and see if they're valid for open, closed, or invalid.
        # Then we compare the set of validations against open/closed and exit if not a valid configuration.
        validation_results = dict()

        any_non_closed = False
        if self.benchmark.params_dict:
            for param, value in self.benchmark.params_dict.items():
                param_validation = self._verify_training_optional_param(self.benchmark.args.model, param, value)
                validation_results[param] = [self.benchmark.args.model, value, param_validation]
                if validation_results[param][2] != PARAM_VALIDATION.CLOSED:
                    any_non_closed = True

        # Add code to verify the other parameters here. Use cluster information and data size commands to verify
        # that the number of processes is appropriate fo the given datasize
        validation_set = set(v[2] for v in validation_results.values())
        if validation_set == {PARAM_VALIDATION.CLOSED}:
            return PARAM_VALIDATION.CLOSED
        elif PARAM_VALIDATION.INVALID in validation_set:
            error_string = "\n\t".join([f"{p} = {v[1]}" for p, v in validation_results.items()])
            self.logger.error(f'\nNot all parameters allowed in closed submission: \n'
                              f'\t{error_string}')
            return PARAM_VALIDATION.INVALID
        else:
            # All open or closed:
            return PARAM_VALIDATION.OPEN

    def _verify_training_optional_param(self, model, param, value):
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

            self.logger.error(f'Invalid parameter {param} for model {model} with value {value}.')
            return PARAM_VALIDATION.INVALID

    def _verify_checkpointing_params(self) -> PARAM_VALIDATION:
        # Rules to Implement:
        # Minimum of 4 processes per physical host during checkpointing
        # For closed, the number of processes can be exactly 8 (subset) or exactly TP x PP x DP from the config
        # For open, the number of processes can be a multiple of the TP x PP from the config

        model = self.benchmark.args.model
        min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES.get(model)
        num_procs = self.benchmark.args.num_processes
        num_hosts = len(self.benchmark.args.hosts)

        validations = set()
        if num_procs / num_hosts >= 4:
            validations.add(PARAM_VALIDATION.CLOSED)
            self.logger.verbose(f'Number of processes per host ({num_procs / num_hosts}) is at least 4.')
        else:
            self.logger.error(f'Number of processes per host ({num_procs / num_hosts}) should be at least 4.')
            validations.add(PARAM_VALIDATION.INVALID)

        if num_procs >= min_procs:
            validations.add(PARAM_VALIDATION.CLOSED)
            self.logger.verbose(f'Number of processes ({num_procs}) is at least {min_procs}.')
        else:
            self.logger.error(f'Number of processes ({num_procs}) should be at least {min_procs}.')
            validations.add(PARAM_VALIDATION.INVALID)

        if num_procs in [ClosedGPUs, LLM_SUBSET_PROCS]:
            self.logger.verbose(f'Number of processes ({num_procs}) is one of {LLM_SUBSET_PROCS} or {ClosedGPUs} in closed submission.')
            validations.add(PARAM_VALIDATION.CLOSED)
        elif self.benchmark.args.closed:
            self.logger.error(f'Number of processes ({num_procs}) should be exactly {LLM_SUBSET_PROCS} or {ClosedGPUs} in closed submission.')
            validations.add(PARAM_VALIDATION.INVALID)
        elif not benchmark.args.closed:
            # num procs should be a multiple of GPUpDP
            dp_instances = num_procs / GPUpDP
            if not dp_instances.is_integer():
                validations.add(PARAM_VALIDATION.INVALID)
                self.logger.error(f'Number of processes ({num_procs}) is not a multiple of {GPUpDP}.')
            else:
                # To get here we've already checked minimum procs, procs per host, and if closed is set
                validations.add(PARAM_VALIDATION.OPEN)
                self.logger.verbose(f'Number of processes ({num_procs}) is a multiple of {GPUpDP}.')

        if validations == {PARAM_VALIDATION.CLOSED}:
            return PARAM_VALIDATION.CLOSED
        elif PARAM_VALIDATION.INVALID in validations:
            return PARAM_VALIDATION.INVALID
        else:
            # Not only closed but no INVALID options == OPEN
            return PARAM_VALIDATION.OPEN


    def _verify_vector_database_params(self):
        # TODO: Implement validation for vector database parameters.
        # Use Cluster Information to verify the size of dataset against the number of clients?
        self.logger.info(f'Need to implement vector database parameter validation.')
        if self.benchmark.args.closed:
            self.logger.error(f'VectorDB is preview only and is not allowed in closed submission.')
            return PARAM_VALIDATION.INVALID

        return PARAM_VALIDATION.CLOSED


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger) -> Tuple[int, int, int]:
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

    If the number of files is greater than MAX_NUM_FILES_TRAIN, use the num_subfolders_train parameters to shard the
    dataset.
    :return:
    """
    required_file_count = 1
    required_subfolders_count = 0

    # Find the amount of memory in the cluster via args or measurements
    measured_total_mem_bytes = cluster_information.info['accumulated_mem_info_bytes']['total']
    if args.client_host_memory_in_gb and args.num_client_hosts:
        # If host memory per client and num clients is provided, we use these values instead of the calculated memory
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
    elif args.clienthost_host_memory_in_gb and not args.num_client_hosts:
        # If we have memory but not clients, we use the number of provided hosts and given memory amount
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 102
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
    else:
        # If no args are provided, measure total memory for given hosts
        total_mem_bytes = measured_total_mem_bytes

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes
    file_size_bytes = dataset_params['num_samples_per_file'] * dataset_params['record_length_bytes']

    min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * args.num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024} MB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')
    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


"""
The results directory structure is as follows:
results_dir:
    <benchmark_name>:
        <command>:
            <subcommand> (Optional)
                <datetime>:
                    run_<run_number> (Optional)
                    
This looks like:
results_dir:
    training:
        unet3d:
            datagen:
                <datetime>:
                    <output_files>
            run:
                <datetime>:
                    run_0:
                        <output_files>
    checkpointing:
        llama3-8b:
            <datetime>:
                <output_files>
"""



def generate_output_location(benchmark, datetime_str=None, **kwargs):
    """
    Generate a standardized output location for benchmark results.

    Output structure follows this pattern:
    RESULTS_DIR:
      <benchmark_name>:
        <command>:
            <subcommand> (Optional)
                <datetime>:
                    run_<run_number> (Optional)

    Args:
        benchmark (Benchmark): benchmark (e.g., 'training', 'vectordb', 'checkpoint')
        datetime_str (str, optional): Datetime string for the run. If None, current datetime is used.
        **kwargs: Additional benchmark-specific parameters:
            - model (str): For training benchmarks, the model name (e.g., 'unet3d', 'resnet50')
            - category (str): For vectordb benchmarks, the category (e.g., 'throughput', 'latency')

    Returns:
        str: The full path to the output location
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    output_location = benchmark.args.results_dir
    if hasattr(benchmark, "run_number"):
        run_number = benchmark.run_number
    else:
        run_number = 0

    # Handle different benchmark types
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

        if benchmark.args.command == "run":
            output_location = os.path.join(output_location, f"run_{run_number}")

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

        if benchmark.args.command == "run-search":
            output_location = os.path.join(output_location, f"run_{run_number}")

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, datetime_str)

    else:
        print(f'The given benchmark is not supported by mlpstorage.rules.generate_output_location()')
        sys.exit(1)

    return output_location


def get_runs_files(results_dir, benchmark_name=None, command=None, logger=None):
    """
    Walk the results_dir location and return a list of dictionaries that represent a single run

    [ { 'benchmark_name': <benchmark_name>,
      'command': <command>,
      'datetime': <datetime>,
      'mlps_metadata_file': <mlps_metadata_file_path>,
      'dlio_summary_json_file': <dlio_summary_json_file_path>,
      'run_number': run_<num>,  #(if applicable)
      'files': [<file_path1>, <file_path2>,...] } ]

    :param results_dir: Base directory containing benchmark results
    :param benchmark_name: Optional filter for specific benchmark name
    :param command: Optional filter for specific command
    :return: List of dictionaries with run information
    """
    if logger is None:
        logger = setup_logging(name='mlpstorage.rules.get_runs_files')

    if not os.path.exists(results_dir):
        logger.warning(f'Results directory {results_dir} does not exist.')
        return []

    runs = []

    # Walk through all directories and files in results_dir
    for root, dirs, files in os.walk(results_dir):
        logger.ridiculous(f'Processing directory: {root}')

        # Look for metadata files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]

        if not metadata_files:
            logger.debug(f'No metadata file found')
            continue
        else:
            logger.debug(f'Found metadata files in directory {root}: {metadata_files}')

        if len(metadata_files) > 1:
            logger.warning(f'Multiple metadata files found in directory {root}. Skipping this directory.')
            continue

        metadata_path = os.path.join(root, metadata_files[0])

        # Find DLIO summary.json file if it exists
        dlio_summary_file = None
        for f in files:
            if f == 'summary.json':
                dlio_summary_file = os.path.join(root, f)
                break

        # Collect all files in this run directory
        run_files = [os.path.join(root, f) for f in files]

        # Create run info dictionary
        run_info = {
            'mlps_metadata_file': metadata_path,
            'dlio_summary_json_file': dlio_summary_file,
            'files': run_files
        }

        runs.append(run_info)

    return runs
