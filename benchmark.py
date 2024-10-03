#!/usr/bin/env python3

import argparse

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
        num_accelerators="Simulated number of accelerators. In multi-host configurations the accelerators will be "
                         "initiated in a round-robin fashion to ensure equal distribution of simulated accelerator "
                         "processes",
        num_client_hosts="Number of participating client hosts. Simulated accelerators will be initiated on these "
                         "hosts in a round-robin fashion",
        client_host_mem_GB="Memory available in the client where the benchmark is run. The dataset needs to be 5x the "
                           "available memory for closed submissions.",
        client_hosts="Comma separated IP addresses of the participating hosts (without spaces). "
                     "eg: '192.168.1.1,192.168.1.2'",
        category="Benchmark category to be submitted."
    )

    parser = argparse.ArgumentParser(description="Script to launch the MLPerf Storage benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    datasize = subparsers.add_parser("datasize")
    datasize.add_argument('--workload', '-w', choices=WORKLOADS, help=help_messages['workload'])
    datasize.add_argument('--accelerator-type', '-g', choices=ACCELERATORS, help=help_messages['accelerator_type'])
    datasize.add_argument('--num-accelerators', '-n', type=int, help=help_messages['num_accelerators'])
    datasize.add_argument('--num-client-hosts', '-c', type=int, help=help_messages['num_client_hosts'])
    datasize.add_argument('--client-host-memory-in-gb', '-m', type=int, help=help_messages['client_host_mem_GB'])

    datagen = subparsers.add_parser("datagen")
    datagen.add_argument('--hosts', '-s', type=str, help=help_messages['client_hosts'])
    datagen.add_argument('--category', '-c', choices=CATEGORIES, help=help_messages['category'])

    return parser.parse_args()


def main():
    print(f'Got to main')


if __name__ == "__main__":
    args = parse_arguments()
    main()
