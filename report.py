import os
import sys
import json
import logging
import argparse
import numpy as np

# final report created by Storage benchmark run
REPORT_FILE = "mlperf_storage_report.json"

# accelerator utilization has to meet AU_THRESHOLD
AU_THRESHOLD = 90

# summary file created by DLIO in the results folder after every run
SUMMARY_FILE = "summary.json"

class StorageReport(object):

    def __init__(self, args):
        # summary file create
        self.result_dir = args.result_dir
        self.multi_host = args.multi_host

    def find_file_path(self, directory):
        found_files = []
        for root, dirs, files in os.walk(directory):
            if SUMMARY_FILE in files:
                found_files.append(os.path.join(root, SUMMARY_FILE))
        return found_files

    def save_data(self, results):
        # Dump statistic counters to files
        # Overall stats
        with open(REPORT_FILE, 'w') as outfile:
            json.dump(results, outfile, indent=4,  default=str)
        logging.info(f"Final report generated: {REPORT_FILE}")


    # read summary for DLIO summary file
    def get_summary(self, summary_file):
        f = open(summary_file)
        summary = json.load(f)
        num_acclerators = summary['num_accelerators']
        host_names =  summary['hostname']
        au = summary['metric']['train_au_mean_percentage']
        throughput_sps = summary['metric']['train_throughput_mean_samples_per_second']
        throughput_mps = summary['metric']['train_io_mean_MB_per_second']
        return (num_acclerators, au, throughput_sps, throughput_mps, host_names)

    # accumulate results from multiple directories in case of multi hosts
    # report benchmark success or failure in case of a single host
    def generate_report(self):
        runs = {}
        summary_files = self.find_file_path(self.result_dir)
        if len(summary_files) == 0:
            logging.error(f"Error: {SUMMARY_FILE} file not found in {self.result_dir}")
            sys.exit(1)
        # report benchmark success or failure in case of a single host
        if not self.multi_host:
            result = {}
            if len(summary_files) > 1:
                logging.error(f"Error: Multiple files found with the same file name {SUMMARY_FILE}")
                sys.exit(1)
            else:
                #report success/failure and return
                metrics = self.get_summary(summary_files[0])
                num_acclerators = metrics[0]
                au = metrics[1]
                throughput_sps = metrics[2]
                throughput_mps = metrics[3]

                status = "succeeded" if float(au) >= AU_THRESHOLD else "failed"
                logging.info("------------------------------")
                logging.info(f"Benchmark {status}")
                logging.info(f"Number of accelerators: {num_acclerators}")
                logging.info(f"Average training throughput: {throughput_sps:.2f} samples/sec({throughput_mps:.2f} MB/sec)")
                logging.info("------------------------------")
                result = {"status": status}
            return result
        # accumulate results from multiple directories in case of multi hosts
        else:
            results={}
            results["overall"] = {}
            results["runs"] = {}
            train_throughput = []
            train_au = []
            for summary_file in summary_files:
                path = summary_file.split("/")
                if len(path) != 4:
                    logging.error(f"Error: Directory structure {summary_file} is not correct. It has be in format result_dir/run(1..n)/host(1..n)/summary.json")
                    sys.exit(1)
                run_name = path[1]
                if run_name not in runs:
                    runs[run_name] = [summary_file]
                else:
                    runs[run_name].append(summary_file)
            host_arr = [len(runs[run_name]) for run_name in runs]
            if len(set(host_arr)) != 1:
                logging.error("Error: Number of participating hosts must be same across all runs")
                sys.exit(1)
            num_hosts = host_arr[0]
            for run_name in runs:
                num_acclerators = []
                train_throughput_sps = []
                train_throughput_mps = []
                host_names = []
                results["runs"][run_name] ={}
                for summary_file in runs[run_name]:
                    summary = self.get_summary(summary_file)
                    au = summary[1]
                    if float(au) < AU_THRESHOLD:
                        logging.error(f"Error: AU value didn't pass the threshold in the run reported by {summary_file}")
                    num_acclerators.append(summary[0])
                    train_throughput_sps.append(summary[1])
                    train_throughput_mps.append(summary[3])
                    host_names.append(summary[4])
                if len(set(host_names)) != len(host_names):
                    logging.warning(f"Warning: Hostnames in results of run {run_name} are not unique")

                #results["runs"][run_name]["num_acclerators"] = num_acclerators
                #results["runs"][run_name]["train_throughput_samples_per_second"] = train_throughput_sps
                #results["runs"][run_name]["train_throughput_MB_per_second"] = train_throughput_mps
                results["runs"][run_name]["train_throughput_samples_per_second"] = np.sum(np.array(train_throughput_sps))
                results["runs"][run_name]["train_throughput_MB_per_second"] = np.sum(np.array(train_throughput_mps))
                results["runs"][run_name]["train_num_accelerators"] = np.sum(np.array(num_acclerators))

            overall_train_throughput_sps = [results["runs"][run_name]["train_throughput_samples_per_second"] for run_name in results["runs"]]
            overall_train_throughput_mps = [results["runs"][run_name]["train_throughput_MB_per_second"] for run_name in results["runs"]]
            overall_train_num_accelerators = [results["runs"][run_name]["train_num_accelerators"] for run_name in results["runs"]]

            if len(set(overall_train_num_accelerators)) != 1:
                logging.error(f"Error: Number of accelerators are different across runs")
                sys.exit(1)
            results["overall"]["num_client_hosts"] = num_hosts
            results["overall"]["num_benchmark_runs"] = len(results["runs"])
            results["overall"]["train_num_accelerators"] =  overall_train_num_accelerators[0]
            results["overall"]["train_throughput_mean_samples_per_second"] = np.mean(overall_train_throughput_sps)
            results["overall"]["train_throughput_stdev_samples_per_second"] = np.std(overall_train_throughput_sps)
            results["overall"]["train_throughput_mean_MB_per_second"] = np.mean(overall_train_throughput_mps)
            results["overall"]["train_throughput_stdev_MB_per_second"] = np.std(overall_train_throughput_mps)
            results["overall"]["train_num_accelerators"] =  overall_train_num_accelerators[0]
            logging.info("------------------------------")
            logging.info(f'Number of client hosts: {results["overall"]["num_client_hosts"]}')
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
    parser.add_argument("-sh", "--multi-host", action="store_true",
            help="If set, multi host results are considered else single host results are considered")
    parser.add_argument("-rg", "--create-report", action="store_true",
        help="If set, result report file generation will be created  ")
    logging.basicConfig(
            format='%(asctime)s %(message)s',
            level=logging.DEBUG,
            datefmt="%Y-%m-%d %H:%M:%S")
    args = parser.parse_args()
    postproc = StorageReport(args)
    results = postproc.generate_report()
    if args.create_report:
        postproc.save_data(results)

if __name__ == '__main__':
    main()
