import os
import sys
import json
import logging
import argparse
import numpy as np
from dateutil import parser

# final report created by Storage benchmark run
REPORT_FILE = "mlperf_storage_report.json"

# summary file created by DLIO in the results folder after every run
SUMMARY_FILE = "summary.json"

# config files containing workload details
CONFIG_OVERRIDES_FILE = "configs/overrides.yaml"

CONFIG_WORKLOAD_FILE = "configs/config.yaml"

# minimum runs required for the submission
REQUIRED_BENCHMARK_RUNS = 5


def find_file_path(directory):
        found_files = []
        for root, dirs, files in os.walk(directory):
            if SUMMARY_FILE in files:
                found_files.append(os.path.join(root, SUMMARY_FILE))
        return found_files

def save_data(results):
    # Dump statistic counters to files
    # Overall stats
    with open(REPORT_FILE, 'w') as outfile:
        json.dump(results, outfile, indent=4,  default=str)
    logging.info(f"Final report generated: {REPORT_FILE}")

def check_unique(list_arg):
    if len(set(list_arg)) == 1:
        return True
    else:
        return False

# read summary for DLIO summary file
def get_summary_details(summary_file):
    f = open(summary_file)
    summary = json.load(f)
    return summary

def get_workload_details(config_file):
    with open(config_file, 'r') as file:
        lines = file.readlines()
    workload_str="workload="
    for line in lines:
        if workload_str in line:
            workload_l = line.split(workload_str)[1].strip()
            workload_details = workload_l.split('_')
            workload = workload_details[0]
            accelerator_type = workload_details[1]
            return workload, accelerator_type
    return "", ""



class StorageReport(object):

    def __init__(self, args):
        # summary file create
        self.result_dir = args.result_dir

    # accumulate results from multiple directories in case of multi hosts
    # report benchmark success or failure in case of a single host
    def generate_report(self):
        runs = {}
        summary_files = find_file_path(self.result_dir)
        if len(summary_files) == 0:
            logging.error(f"Error: {SUMMARY_FILE} file not found in {self.result_dir}")
            sys.exit(1)

        # accumulate results from multiple directories in case of multi hosts
        results={}
        results["overall"] = {}
        results["runs"] = {}
        train_throughput = []
        train_au = []
        for summary_file in summary_files:
            run_path = os.path.relpath(summary_file, self.result_dir)
            run_dir = run_path.split("/")
            if len(run_dir) != 2:
                logging.error(f"Error: Directory structure {summary_file} is not correct. It has be in format result_dir/run(1..n)/summary.json")
                sys.exit(1)
            run_name = run_dir[0]
            runs[run_name] = summary_file
        if len(runs) != REQUIRED_BENCHMARK_RUNS:
            logging.error(f"Error: Results are reported only for {len(runs)} runs. {REQUIRED_BENCHMARK_RUNS} runs are required.")
            sys.exit(1)
        for run_name in runs:
            models = []
            num_acclerators = []
            train_throughput_sps = []
            train_throughput_mps = []
            host_names = []
            num_files_train = []
            num_samples_per_file = []
            start_host_timestamp = []
            results["runs"][run_name] ={}

            summary_file = runs[run_name]
            config_file = os.path.join(os.path.dirname(summary_file), CONFIG_OVERRIDES_FILE)
            model, accelerator = get_workload_details(config_file)
            if not model or not accelerator:
                logging.error("workload missing in the config file", CONFIG_OVERRIDES_FILE)
                sys.exit(1)

            summary = get_summary_details(runs[run_name])
            au = summary['metric']['train_au_mean_percentage']
            if summary['metric']['train_au_meet_expectation'] == "fail":
                    logging.error(f"Error: AU value {au} didn't pass the threshold in the run reported by {summary_file}")
                    sys.exit(1)
            num_acclerators.append(summary['num_accelerators'])
            train_throughput_sps.append(summary['metric']['train_throughput_mean_samples_per_second'])
            train_throughput_mps.append(summary['metric']['train_io_mean_MB_per_second'])
            num_files_train.append(summary['num_files_train'])
            num_samples_per_file.append(summary['num_samples_per_file'])

            results["runs"][run_name]["train_throughput_samples_per_second"] = np.sum(np.array(train_throughput_sps))
            results["runs"][run_name]["train_throughput_MB_per_second"] = np.sum(np.array(train_throughput_mps))
            results["runs"][run_name]["train_num_accelerators"] = np.sum(np.array(num_acclerators))
            results["runs"][run_name]["model"] = model
            results["runs"][run_name]["accelerator"] = accelerator
            results["runs"][run_name]["num_files_train"] = num_files_train[0]
            results["runs"][run_name]["num_samples_per_file"] = num_samples_per_file[0]


        overall_train_throughput_sps = [results["runs"][run_name]["train_throughput_samples_per_second"] for run_name in results["runs"]]
        overall_train_throughput_mps = [results["runs"][run_name]["train_throughput_MB_per_second"] for run_name in results["runs"]]
        overall_model = [results["runs"][run_name]["model"] for run_name in results["runs"]]
        overall_accelerator = [results["runs"][run_name]["accelerator"] for run_name in results["runs"]]
        overall_train_num_accelerators = [results["runs"][run_name]["train_num_accelerators"] for run_name in results["runs"]]
        overall_num_files_train = [results["runs"][run_name]["num_files_train"] for run_name in results["runs"]]
        overall_num_samples_per_file = [results["runs"][run_name]["num_samples_per_file"] for run_name in results["runs"]]

        if not check_unique(overall_model):
            logging.error(f"Error: The model name is different across runs")
            sys.exit(1)
        if not check_unique(overall_accelerator):
            logging.error(f"Error: The model name is different across runs")
            sys.exit(1)
        if not check_unique(overall_train_num_accelerators):
            logging.error(f"Error: The number of accelerators is different across runs")
            sys.exit(1)
        if not check_unique(overall_num_files_train):
            logging.error(f"Error: The number of training files is different across runs")
            sys.exit(1)
        if not check_unique(overall_num_samples_per_file):
            logging.error(f"Error: The number of samples per file is different across runs")
            sys.exit(1)

        results["overall"]["model"] = overall_model[0]
        results["overall"]["accelerator"] = overall_accelerator[0]
        results["overall"]["num_benchmark_runs"] = len(results["runs"])
        results["overall"]["train_num_accelerators"] =  overall_train_num_accelerators[0]
        results["overall"]["num_files_train"] = overall_num_files_train[0]
        results["overall"]["num_samples_per_file"] = overall_num_samples_per_file[0]
        results["overall"]["train_throughput_mean_samples_per_second"] = np.mean(overall_train_throughput_sps)
        results["overall"]["train_throughput_stdev_samples_per_second"] = np.std(overall_train_throughput_sps)
        results["overall"]["train_throughput_mean_MB_per_second"] = np.mean(overall_train_throughput_mps)
        results["overall"]["train_throughput_stdev_MB_per_second"] = np.std(overall_train_throughput_mps)
        logging.info("------------------------------")
        logging.info(f'Model: {results["overall"]["model"]}')
        logging.info(f'Accelerator: {results["overall"]["accelerator"]}')
        logging.info(f'Number of benchmark runs: {results["overall"]["num_benchmark_runs"]}')
        logging.info(f'Overall number of accelerators: {results["overall"]["train_num_accelerators"]}')
        logging.info(f'Overall Training Throughput (samples/second): {results["overall"]["train_throughput_mean_samples_per_second"]:.2f} ({results["overall"]["train_throughput_stdev_samples_per_second"]})')
        logging.info(f'Overall Training Throughput (MB/second): {results["overall"]["train_throughput_mean_MB_per_second"]:.2f} ({results["overall"]["train_throughput_stdev_MB_per_second"]})')
        logging.info("------------------------------")
        return results

def main():
    """
    The main method to start the benchmark runtime.
    """
    parser = argparse.ArgumentParser(description='Storage report generator')
    parser.add_argument("-rd", "--result-dir", type=str, default="",
                        help="Location to the results directory of a benchmark run which contains summary.json")
    logging.basicConfig(
            format='%(asctime)s %(message)s',
            level=logging.DEBUG,
            datefmt="%Y-%m-%d %H:%M:%S")
    args = parser.parse_args()
    postproc = StorageReport(args)
    results = postproc.generate_report()
    save_data(results)

if __name__ == '__main__':
    main()
