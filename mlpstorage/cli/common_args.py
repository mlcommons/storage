"""
Common CLI arguments and help messages shared across benchmarks.

This module contains:
- Help message definitions
- Universal argument functions
- MPI argument group
- Validation utilities
"""

from mlpstorage.config import (
    CHECKPOINT_RANKS_STRINGS, MODELS, ACCELERATORS, DEFAULT_HOSTS,
    LLM_MODELS_STRINGS, MPI_CMDS, EXEC_TYPE, DEFAULT_RESULTS_DIR,
    VECTOR_DTYPES, DISTRIBUTIONS
)


# Help messages dictionary - shared across all argument builders
HELP_MESSAGES = {
    # General help messages
    'sub_commands': "Select a subcommand for the benchmark.",
    'model': (
        "Model to emulate. A specific model defines the sample size, sample container format, and data \n"
        "rates for each supported accelerator."
    ),
    'accelerator_type': (
        "Accelerator to simulate for the benchmark. A specific accelerator defines the data access "
        "sizes and rates for each supported workload"
    ),
    'num_accelerators_datasize': (
        "Max number of simulated accelerators. In multi-host configurations the accelerators "
        "will be initiated in a round-robin fashion to ensure equal distribution of "
        "simulated accelerator processes"
    ),
    'num_accelerators_run': (
        "Number of simulated accelerators. In multi-host configurations the accelerators "
        "will be initiated in a round-robin fashion to ensure equal distribution of "
        "simulated accelerator processes"
    ),
    'num_accelerators_datagen': (
        "Number of parallel processes to use for dataset generation. Processes will be "
        "initiated in a round-robin fashion across the configured client hosts"
    ),
    'num_client_hosts': (
        "Number of participating client hosts. Simulated accelerators will be initiated on these "
        "hosts in a round-robin fashion"
    ),
    'client_host_mem_GB': (
        "Memory available in the client where the benchmark is run. The dataset needs to be 5x the "
        "available memory for closed submissions."
    ),
    'client_hosts': (
        "Space-separated list of IP addresses or hostnames of the participating hosts. "
        "\nExample: '--hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'. Slots can "
        "be specified by appending ':<num_slots>' to a hostname like so: '--hosts host1:2 host2:2'. This "
        "example will run 2 accelerators on each host. If slots are not specified the number of processes "
        "will be equally distributed across the hosts with any remainder being distributed evenly on the "
        "remaining hosts in the order they are listed."
    ),
    'category': "Benchmark category to be submitted.",
    'results_dir': "Directory where the benchmark results will be saved.",
    'params': (
        "Additional parameters to be passed to the benchmark. These will override the config file. "
        "\nFor a closed submission only a subset of params are supported. "
        "\nMultiple values allowed in the form: "
        "\n    --params key1=value1 key2=value2 key3=value3"
    ),
    'datasize': (
        "The datasize command calculates the number of samples needed for a given workload, accelerator type,"
        " number of accelerators, and client host memory."
    ),
    'training_datagen': (
        "The datagen command generates a dataset for a given workload and number of parallel generation "
        "processes."
    ),
    'run_benchmark': "Run the benchmark with the specified parameters.",
    'configview': "View the final config based on the specified options.",
    'reportgen': "Generate a report from the benchmark results.",

    # Checkpoint folder is used for training and checkpointing
    'checkpoint_folder': "Location for checkpoint files for training or checkpointing workloads",

    # Checkpointing help messages
    'checkpoint_run': "The checkpoint command executes checkpoint saves and restores for a given model.",
    'llm_model': (
        "The model & size to be emulated for checkpointing. The selection will dictate the TP, PP, & DP "
        "\nsizes as well as the size of the checkpoint. "
        "\nAvailable LLM Models: "
        f"\n    {LLM_MODELS_STRINGS}"
    ),
    'num_checkpoints': "The number of checkpoints to be executed.",
    'num_checkpoint_accelerators': (
        f"The number of accelerators to emulate for the checkpoint task. Each LLM Model "
        f"\ncan be executed with the following accelerator counts: "
        f"\n    {CHECKPOINT_RANKS_STRINGS}"
    ),
    'deepspeed_zero_level': (
        "The DeepSpeed Zero level. \nSupported options: "
        "\n    0 = disabled, "
        "\n    1 = Optimizer Partitioning, "
        "\n    2 = Gradient partitioning, "
        "\n    3 = Model Parameter Partitioning"
    ),
    'checkpoint_datasize': (
        "The datasize command calculates the total amount of writes for a given command and an estimate "
        "of the required memory."
    ),
    'checkpoint_subset': (
        "Run the checkpoint in 'Subset' mode. This mode only runs on a subset of hosts. eg, for large "
        "models that required hundreds of processes to do an entire checkpoint, subset mode enables "
        "using fewer processes and only doing part of the checkpoint. This is used in the Submissions to "
        "represent a single 8-GPU node writing to local storage."
    ),

    # VectorDB help messages
    'db_ip_address': "IP address of the VectorDB instance. If not provided, a local VectorDB instance will be used.",
    'db_port': "Port number of the VectorDB instance.",
    'db_collection': "Collection name for the VectorDB instance.",
    'dimension': "Dimensionality of the vectors.",
    'num_shards': "Number of shards for the collection. Recommended is 1 for every 1 Million vectors",
    'vector_dtype': f"Data type of the vectors. Supported options: {VECTOR_DTYPES}",
    'num_vectors': "Number of vectors to be inserted into the collection.",
    'distribution': f"Distribution of the vectors. Supported options: {DISTRIBUTIONS}",
    'vdb_datagen_batch_size': "Batch size for data insertion.",
    'vdb_datagen_chunk_size': "Number of vectors to generate in each insertion chunk. Tune for memory management.",
    'vdb_run_search': "Run the VectorDB Search benchmark with the specified parameters.",
    'vdb_datagen': "Generate a dataset for the VectorDB benchmark.",
    'vdb_report_count': "Number of batches between print statements",
    'num_query_processes': "Number of parallel processes to use for query execution.",
    'query_batch_size': "Number of vectors to query in each batch (per process).",

    # Reports help messages
    'output_dir': "Directory where the benchmark report will be saved.",
    'config_file': "Path to YAML file with argument overrides that will be applied after CLI arguments",

    # MPI help messages
    'mpi_bin': f"Execution type for MPI commands. Supported options: {MPI_CMDS}",
    'exec_type': f"Execution type for benchmark commands. Supported options: {list(EXEC_TYPE)}",
}

# Program descriptions
PROGRAM_DESCRIPTIONS = {
    'training': "Run the MLPerf Storage training benchmark",
    'checkpointing': "Run the MLPerf Storage checkpointing benchmark",
    'vectordb': "Run the MLPerf Storage Preview of a VectorDB benchmark (not available in closed submissions)",
    'kvcache': "Run the KV Cache benchmark for LLM inference storage testing (preview)",
    'reports': "Generate reports from benchmark results",
    'history': "View and manage benchmark history",
}


def add_universal_arguments(parser):
    """Add arguments common to all benchmarks and commands.

    Args:
        parser: Argparse parser to add arguments to.
    """
    standard_args = parser.add_argument_group("Standard Arguments")
    standard_args.add_argument(
        '--results-dir', '-rd',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=HELP_MESSAGES['results_dir']
    )
    standard_args.add_argument(
        '--loops',
        type=int,
        default=1,
        help="Number of times to run the benchmark"
    )
    standard_args.add_argument(
        '--config-file', '-c',
        type=str,
        help="Path to YAML file with argument overrides"
    )

    # Create a mutually exclusive group for closed/open options
    submission_group = standard_args.add_mutually_exclusive_group()
    submission_group.add_argument(
        "--open",
        action="store_false",
        dest="closed",
        default=False,
        help="Run as an open submission"
    )
    submission_group.add_argument(
        "--closed",
        action="store_true",
        help="Run as a closed submission"
    )

    output_control = parser.add_argument_group("Output Control")
    output_control.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    output_control.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode"
    )
    output_control.add_argument(
        "--stream-log-level",
        type=str,
        default="INFO"
    )
    output_control.add_argument(
        "--allow-invalid-params", "-aip",
        action="store_true",
        help="Do not fail on invalid parameters."
    )

    view_only_args = parser.add_argument_group("View Only")
    view_only_args.add_argument(
        "--what-if",
        action="store_true",
        help="View the configuration that would execute and the associated command."
    )

    validation_args = parser.add_argument_group("Validation")
    validation_args.add_argument(
        "--verify-lockfile",
        type=str,
        metavar="PATH",
        help="Validate installed packages against lockfile before benchmark execution",
    )
    validation_args.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation (MPI, SSH, DLIO checks). Useful for debugging.",
    )


def add_mpi_arguments(parser):
    """Add MPI-related arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    mpi_options = parser.add_argument_group("MPI")
    mpi_options.add_argument(
        '--mpi-bin',
        choices=MPI_CMDS,
        default="mpirun",
        help=HELP_MESSAGES['mpi_bin']
    )
    mpi_options.add_argument(
        '--oversubscribe',
        action="store_true"
    )
    mpi_options.add_argument(
        '--allow-run-as-root',
        action="store_true"
    )
    mpi_options.add_argument(
        '--mpi-params',
        nargs="+",
        type=str,
        action="append",
        help="Other MPI parameters that will be passed to MPI"
    )


def add_host_arguments(parser, required=False):
    """Add host-related arguments common to distributed benchmarks.

    Args:
        parser: Argparse parser to add arguments to.
        required: Whether hosts argument is required.
    """
    parser.add_argument(
        '--hosts', '-s',
        nargs="+",
        default=DEFAULT_HOSTS if not required else None,
        required=required,
        help=HELP_MESSAGES['client_hosts']
    )


def add_dlio_arguments(parser):
    """Add DLIO-related arguments.

    Args:
        parser: Argparse parser to add arguments to.
    """
    parser.add_argument(
        '--dlio-bin-path', '-dp',
        type=str,
        help="Path to DLIO binary. Default is the same as mlpstorage binary path"
    )
    parser.add_argument(
        '--params', '-p',
        nargs="+",
        type=str,
        action="append",
        help=HELP_MESSAGES['params']
    )
