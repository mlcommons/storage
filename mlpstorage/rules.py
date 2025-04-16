from .config import MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, MAX_NUM_FILES_TRAIN


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


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger):
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
    required_file_count = 0
    required_subfolders_count = 0
    required_samples_per_file = 0

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
    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024} MB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')
    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.verbose(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return required_file_count, required_subfolders_count, required_samples_per_file
