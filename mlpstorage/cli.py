import argparse
import sys


from mlpstorage.config import MIN_RANKS_STR, MODELS, ACCELERATORS, DEFAULT_HOSTS, CATEGORIES, LLM_MODELS, MPI_CMDS, \
    EXEC_TYPE, DEFAULT_RESULTS_DIR, EXIT_CODE, INDEX_TYPES, VECTOR_DTYPES, DISTRIBUTIONS

# TODO: Get rid of this now that I'm not repeating arguments for different subparsers?
help_messages = dict(
    # General help messages
    sub_commands="Select a subcommand for the benchmark.",
    model="Model to emulate. A specific model defines the sample size, sample container format, and data "
          "rates for each supported accelerator.",
    accelerator_type="Accelerator to simulate for the benchmark. A specific accelerator defines the data access "
                     "sizes and rates for each supported workload",
    num_accelerators_datasize="Max number of simulated accelerators. In multi-host configurations the accelerators "
                              "will be initiated in a round-robin fashion to ensure equal distribution of "
                              "simulated accelerator processes",
    num_accelerators_run="Number of simulated accelerators. In multi-host configurations the accelerators "
                         "will be initiated in a round-robin fashion to ensure equal distribution of "
                         "simulated accelerator processes",
    num_accelerators_datagen="Number of parallel processes to use for dataset generation. Processes will be "
                             "initiated in a round-robin fashion across the configured client hosts",
    num_client_hosts="Number of participating client hosts. Simulated accelerators will be initiated on these "
                     "hosts in a round-robin fashion",
    client_host_mem_GB="Memory available in the client where the benchmark is run. The dataset needs to be 5x the "
                       "available memory for closed submissions.",
    client_hosts="Space-separated list of IP addresses or hostnames of the participating hosts. "
                 "Example: '--hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'",
    category="Benchmark category to be submitted.",
    results_dir="Directory where the benchmark results will be saved.",
    params="Additional parameters to be passed to the benchmark. These will override the config file. For a closed "
           "submission only a subset of params are supported. Multiple values allowed in the form: "
           "--params key1=value1 key2=value2 key3=value3",
    datasize="The datasize command calculates the number of samples needed for a given workload, accelerator type,"
             " number of accelerators, and client host memory.",
    training_datagen="The datagen command generates a dataset for a given workload and number of parallel generation "
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
                                f"\n{MIN_RANKS_STR}",

    # VectorDB help messages
    db_ip_address=f"IP address of the VectorDB instance. If not provided, a local VectorDB instance will be used.",
    db_port=f"Port number of the VectorDB instance.",
    db_collection=f"Collection name for the VectorDB instance.",
    dimension=f"Dimensionality of the vectors.",
    num_shards=f"Number of shards for the collection. Recommended is 1 for every 1 Million vectors",
    vector_dtype=f"Data type of the vectors. Supported options: {VECTOR_DTYPES}",
    num_vectors=f"Number of vectors to be inserted into the collection.",
    distribution=f"Distribution of the vectors. Supported options: {DISTRIBUTIONS}",
    vdb_datagen_batch_size=f"Batch size for data insertion.",
    vdb_datagen_chunk_size="Number of vectors to generate in each insertion chunk. Tune for memory management.",

    vdb_run_search="Run the VectorDB Search benchmark with the specified parameters.",
    vdb_datagen="Generate a dataset for the VectorDB benchmark.",
    vdb_report_count="Number of batches between print statements",
    num_query_processes=f"Number of parallel processes to use for query execution.",
    query_batch_size=f"Number of vectors to query in each batch (per process).",

    # Reports help messages
    output_dir=f"Directory where the benchmark report will be saved.",

    # MPI help messages
    mpi_bin=f"Execution type for MPI commands. Supported options: {MPI_CMDS}",
    exec_type=f"Execution type for benchmark commands. Supported options: {list(EXEC_TYPE)}",
)

prog_descriptions = dict(
    training="Run the MLPerf Storage training benchmark",
    checkpointing="Run the MLPerf Storage checkpointing benchmark",
    vectordb="Run the MLPerf Storage Preview of a VectorDB benchmark (not available in closed submissions)",
)

def parse_arguments():
    # Many of the help messages are shared between the subparsers. This dictionary prevents rewriting the same messages
    # in multiple places.
    parser = argparse.ArgumentParser(description="Script to launch the MLPerf Storage benchmark")
    sub_programs = parser.add_subparsers(dest="program", required=True)
    sub_programs.required = True

    training_parsers = sub_programs.add_parser("training", description=prog_descriptions['training'],
                                               help="Training benchmark options")
    checkpointing_parsers = sub_programs.add_parser("checkpointing", description=prog_descriptions['checkpointing'],
                                                    help="Checkpointing benchmark options")
    vectordb_parsers = sub_programs.add_parser("vectordb", description=prog_descriptions['vectordb'],
                                               help="VectorDB benchmark options")
    reports_parsers = sub_programs.add_parser("reports", help="Generate a report from benchmark results")
    history_parsers = sub_programs.add_parser("history", help="Display benchmark history")

    sub_programs_map = dict(training=training_parsers,
                            checkpointing=checkpointing_parsers,
                            vectordb=vectordb_parsers,
                            reports=reports_parsers,
                            history=history_parsers
                            )

    add_training_arguments(training_parsers)
    add_checkpointing_arguments(checkpointing_parsers)
    add_vectordb_arguments(vectordb_parsers)
    add_reports_arguments(reports_parsers)
    add_history_arguments(history_parsers)

    for _parser in sub_programs_map.values():
        add_universal_arguments(_parser)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if len(sys.argv) == 2 and sys.argv[1] in sub_programs_map.keys():
        sub_programs_map[sys.argv[1]].print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


# These are used by the history tracker to know if logging needs to be updated.
logging_options = ['debug', 'verbose', 'stream_log_level']

def add_universal_arguments(parser):
    standard_args = parser.add_argument_group("Standard Arguments")
    standard_args.add_argument('--results-dir', '-rd', type=str, default=DEFAULT_RESULTS_DIR, help=help_messages['results_dir'])

    # Create a mutually exclusive group for closed/open options
    submission_group = standard_args.add_mutually_exclusive_group()
    submission_group.add_argument("--open", action="store_false", dest="closed", default=False,
                                  help="Run as an open submission")
    submission_group.add_argument("--closed", action="store_true", help="Run as a closed submission")

    output_control = parser.add_argument_group("Output Control")
    output_control.add_argument("--debug", action="store_true", help="Enable debug mode")
    output_control.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    output_control.add_argument("--version", action="version", version="%(prog)s 0.1")
    output_control.add_argument("--stream-log-level", type=str, default="INFO",)

    output_control.add_argument("--allow-invalid-params", "-aip", action="store_true",
                        help="Do not fail on invalid parameters.")

    view_only_args = parser.add_argument_group("View Only")
    view_only_args.add_argument("--what-if", action="store_true", help="View the configuration that would execute and "
                                                                       "the associated command.")


def add_mpi_group(parser):
    mpi_options = parser.add_argument_group("MPI")
    mpi_options.add_argument('--mpi-bin', choices=MPI_CMDS, default="mpiexec", help=help_messages['mpi_bin'])
    mpi_options.add_argument('--oversubscribe', action="store_true")
    mpi_options.add_argument('--allow-run-as-root', action="store_true")


def add_training_arguments(training_parsers):
    training_subparsers = training_parsers.add_subparsers(dest="command", required=True)
    training_parsers.required = True

    datasize = training_subparsers.add_parser("datasize", help=help_messages['datasize'])
    datagen = training_subparsers.add_parser("datagen", help=help_messages['training_datagen'])
    run_benchmark = training_subparsers.add_parser("run", help=help_messages['run_benchmark'])
    configview = training_subparsers.add_parser("configview", help=help_messages['configview'])
    reportgen = training_subparsers.add_parser("reportgen", help=help_messages['reportgen'])

    for _parser in [datasize, datagen, run_benchmark]:
        _parser.add_argument('--hosts', '-s', nargs="+", default=DEFAULT_HOSTS, help=help_messages['client_hosts'])
        _parser.add_argument('--model', '-m', choices=MODELS, required=True, help=help_messages['model'])

        # TODO: Add exclusive group for memory or auto-scaling
        # For 'datagen' this should be used to ensure enough memory exists to do the generation. The if statement
        #  prevents it from being use today but change when we add the capability
        if _parser != datagen:
            _parser.add_argument('--client-host-memory-in-gb', '-cm', type=int, required=True, help=help_messages['client_host_mem_GB'])

        _parser.add_argument('--exec-type', '-et', type=EXEC_TYPE, choices=list(EXEC_TYPE), default=EXEC_TYPE.MPI, help=help_messages['exec_type'])

        add_mpi_group(_parser)

    datagen.add_argument('--num-processes', '-np', type=int, required=True, help=help_messages['num_accelerators_datagen'])
    datasize.add_argument('--max-accelerators', '-ma', type=int, required=True, help=help_messages['num_accelerators_datasize'])
    run_benchmark.add_argument('--num-accelerators', '-na', type=int, required=True, help=help_messages['num_accelerators_run'])
    configview.add_argument('--num-accelerators', '-na', type=int, required=True, help=help_messages['num_accelerators_run'])

    for _parser in [datasize, run_benchmark]:
        _parser.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, required=True, help=help_messages['accelerator_type'])
        _parser.add_argument('--num-client-hosts', '-nc', type=int, required=True, help=help_messages['num_client_hosts'])

    for _parser in [datasize, datagen, run_benchmark, configview]:
        _parser.add_argument('--ssh-username', '-u', type=str, help="Username for SSH for system information collection")
        _parser.add_argument('--params', '-p', nargs="+", type=str, help=help_messages['params'])

    for _parser in [datasize, datagen, run_benchmark, configview, reportgen]:
        _parser.add_argument("--data-dir", '-dd', type=str, help="Filesystem location for data")
        add_universal_arguments(_parser)


def add_checkpointing_arguments(checkpointing_parsers):
    # Checkpointing

    checkpointing_subparsers = checkpointing_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    checkpointing_parsers.required = True

    # Add specific checkpointing benchmark options here
    checkpoint = checkpointing_subparsers.add_parser('checkpoint', help=help_messages['checkpoint'])
    checkpoint.add_argument('--llm-model', '-lm', choices=LLM_MODELS, help=help_messages['llm_model'])
    checkpoint.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])
    checkpoint.add_argument('--num-accelerators', '-na', type=int, help=help_messages['num_checkpoint_accelerators'])

    recovery = checkpointing_subparsers.add_parser('recovery', help=help_messages['recovery'])


def add_vectordb_arguments(vectordb_parsers):
    # VectorDB Benchmark
    vectordb_subparsers = vectordb_parsers.add_subparsers(dest="command", required=True, help="sub_commands")
    vectordb_parsers.required = True

    datagen = vectordb_subparsers.add_parser('datagen', help=help_messages['vdb_datagen'])
    run_search = vectordb_subparsers.add_parser('run-search', help=help_messages['vdb_run_search'])

    for _parser in [datagen, run_search]:
        _parser.add_argument('--host', '-s', type=str, default="127.0.0.1", help=help_messages['db_ip_address'])
        _parser.add_argument('--port', '-p', type=int, default=19530, help=help_messages['db_port'])
        _parser.add_argument('--config')
        _parser.add_argument('--collection', type=str, help=help_messages['db_collection'])

    # Datagen specific arguments
    datagen.add_argument('--dimension', type=int, default=1536, help=help_messages['dimension'])
    datagen.add_argument('--num-shards', type=int, default=1, help=help_messages['num_shards'])
    datagen.add_argument('--vector-dtype', choices=VECTOR_DTYPES, default="FLOAT_VECTOR", help=help_messages['vector_dtype'])
    datagen.add_argument('--num-vectors', type=int, default=1_000_000, help=help_messages['num_vectors'])
    datagen.add_argument('--distribution', choices=DISTRIBUTIONS, default="uniform", help=help_messages['distribution'])
    datagen.add_argument('--batch-size', type=int, default=1_000, help=help_messages['vdb_datagen_batch_size'])
    datagen.add_argument('--chunk-size', type=int, default=10_000, help=help_messages['vdb_datagen_chunk_size'])
    datagen.add_argument("--force", action="store_true", help="Force recreate collection if it exists")

    # Add specific VectorDB benchmark options here
    run_search.add_argument('--num-query-processes', type=int, default=1, help=help_messages['num_query_processes'])
    run_search.add_argument('--batch-size', type=int, default=1, help=help_messages['query_batch_size'])
    run_search.add_argument('--report-count', type=int, default=100, help=help_messages['vdb_report_count'])

    end_group = run_search.add_argument_group("Provide an end condition of runtime (in seconds) or total number of "
                                              "queries to execute. The default is to run for 60 seconds")
    end_condition = end_group.add_mutually_exclusive_group()
    end_condition.add_argument("--runtime", type=int, help="Run for a specific duration in seconds")
    end_condition.add_argument("--queries", type=int, help="Run for a specific number of queries")

    for _parser in [datagen, run_search]:
        add_universal_arguments(_parser)


def add_reports_arguments(reports_parsers):
    # Reporting

    reports_subparsers = reports_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    reports_parsers.required = True

    reportgen = reports_subparsers.add_parser('reportgen', help=help_messages['reportgen'])

    reportgen.add_argument('--output-dir', type=str, help=help_messages['output_dir'])
    reportgen.add_argument('--remove-deleted', action="store_true", help="Remove deleted tests")
    add_universal_arguments(reportgen)


def add_history_arguments(history_parsers):
    # History
    history_subparsers = history_parsers.add_subparsers(dest="command", required=True, help="Sub-commands")
    history_parsers.required = True

    history = history_subparsers.add_parser('show', help="Show command history")
    history.add_argument('--limit', '-n', type=int, help="Limit to the N most recent commands")
    history.add_argument('--id', '-i', type=int, help="Show a specific command by ID")

    rerun = history_subparsers.add_parser('rerun', help="Re-run a command from history")
    rerun.add_argument('rerun_id', type=int, help="ID of the command to re-run")

    for _parser in [history, rerun]:
        add_universal_arguments(_parser)


def validate_args(args):
    error_messages = []
    # Add generic validations here. Workload specific validation is in the Benchmark classes

    if error_messages:
        for msg in error_messages:
            print(msg)

        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)


def update_args(args):
    """
    This method is an interface between the CLI and the benchmark class.
    """
    for arg in ['num_processes', 'num_accelerators', 'max_accelerators']:
        if hasattr(args, arg):
            setattr(args, 'num_processes', int(getattr(args, arg)))
            break

    if hasattr(args, 'runtime') and hasattr(args, 'queries'):
        if not args.runtime and not args.queries:
            args.runtime = 60  # Default runtime if not provided


if __name__ == "__main__":
    args = parse_arguments()
    import pprint
    pprint.pprint(vars(args))

