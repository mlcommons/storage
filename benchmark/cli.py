from .config import MIN_RANKS_STR, MODELS, ACCELERATORS, DEFAULT_HOSTS, CATEGORIES, LLM_MODELS


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