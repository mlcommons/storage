#!/usr/bin/env python3

import argparse
import sys

# Define constants:
COSMOFLOW = "cosmoflow"
RESNET = "resnet50"
UNET = "unet3d"
WORKLOADS = [COSMOFLOW, RESNET, UNET]

H100 = "h100"
A100 = "a100"
ACCELERATORS = [H100, A100]

OPEN = "open"
CLOSED = "closed"
CATEGORIES = [OPEN, CLOSED]

# Capturing TODO Items:
# Change parameters so that each sub parser uses the same flags. For example, datasize uses -c for num-client-hosts
#   but datagen uses -c for category
#
# Configure datasize to collect the memory information from the hosts instead of getting a number of hosts for the
#   calculation


def parse_arguments():
    # Many of the help messages are shared between the subparsers. This dictionary prevents rewriting the same messages
    # in multiple places.
    help_messages = dict(
        workload="Workload to emulate. A specific workload defines the sample size, sample container format, and data "
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
        param="Additional parameters to be passed to the benchmark. These will override the config file. For a closed "
              "submission only a subset of params are supported.",
        datasize="The datasize command calculates the number of samples needed for a given workload, accelerator type,"
                 " number of accelerators, and client host memory.",
        datagen="The datagen command generates a dataset for a given workload and number of parallel generation "
                "processes.",
        run_benchmark="Run the benchmark with the specified parameters.",
        configview="View the final config based on the specified options.",
        reportgen="Generate a report from the benchmark results.",
    )

    parser = argparse.ArgumentParser(description="Script to launch the MLPerf Storage benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    datasize = subparsers.add_parser("datasize", description=help_messages['datasize'])
    datasize.add_argument('--workload', '-w', choices=WORKLOADS, help=help_messages['workload'])
    datasize.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    datasize.add_argument('--num-accelerators', '-n', type=int, help=help_messages['num_accelerators_datasize'])
    datasize.add_argument('--num-client-hosts', '-c', type=int, help=help_messages['num_client_hosts'])
    datasize.add_argument('--client-host-memory-in-gb', '-m', type=int, help=help_messages['client_host_mem_GB'])

    datagen = subparsers.add_parser("datagen", description=help_messages['datagen'])
    datagen.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])
    datagen.add_argument('--category', '-c', choices=CATEGORIES, help=help_messages['category'])
    datagen.add_argument('--workload', '-w', choices=WORKLOADS, help=help_messages['workload'])
    datagen.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    datagen.add_argument('--num-accelerators', '-n', type=int, help=help_messages['num_accelerators_datagen'])
    datagen.add_argument('--results-dir', '-r', type=str, help=help_messages['results_dir'])
    datagen.add_argument('--param', '-p', action="append", nargs="+", type=str, help=help_messages['param'])

    run_benchmark = subparsers.add_parser("run", description=help_messages['run_benchmark'])
    run_benchmark.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])
    run_benchmark.add_argument('--category', '-c', choices=CATEGORIES, help=help_messages['category'])
    run_benchmark.add_argument('--workload', '-w', choices=WORKLOADS, help=help_messages['workload'])
    run_benchmark.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    run_benchmark.add_argument('--num-accelerators', '-n', type=int, help=help_messages['num_accelerators_datasize'])
    run_benchmark.add_argument('--results-dir', '-r', type=str, help=help_messages['results_dir'])
    run_benchmark.add_argument('--param', '-p', action="append", nargs="+", type=str, help=help_messages['param'])

    configview = subparsers.add_parser("configview", description=help_messages['configview'])
    configview.add_argument('--workload', '-w', choices=WORKLOADS, help=help_messages['workload'])
    configview.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    configview.add_argument('--param', '-p', action="append", nargs="+", type=str, help=help_messages['param'])

    reportgen = subparsers.add_parser("reportgen", description=help_messages['reportgen'])
    reportgen.add_argument('--results-dir', '-r', type=str, help=help_messages['results_dir'])

    return parser.parse_args()


def validate_args(args):
    error_messages = []
    # Add code here for validation processes. We do not need to validate an option is in a list as the argparse
    #  option "choices" accomplishes this for us.

    if error_messages:
        for msg in error_messages:
            print(msg)

    sys.exit(1)


def main(args):
    print(f'Got to main')
    import pprint
    pprint.pprint(vars(args))

    validate_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
