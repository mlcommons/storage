"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import re
import json
import logging
import argparse
import pandas as pd
from dlio_benchmark.utils.utility import str2bool
from statistics import mean, median, stdev, quantiles
from dlio_benchmark.utils.config import ConfigArguments, LoadConfig
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import yaml 
import glob
import numpy as np


class DLIOPostProcessor:
    def __init__(self, args) -> None:
        self.name = args.name
        self.outdir = args.output_folder
        self.comm_size = args.num_proc
        self.epochs = args.epochs
        self.epochs_list = [str(e) for e in range(1, self.epochs + 1)]

        self.do_eval = args.do_eval
        self.do_checkpoint = args.do_checkpoint

        self.batch_size = args.batch_size
        self.batch_size_eval = args.batch_size_eval
        self.iotrace = None
        self.per_epoch_stats = None

        self.verify_and_load_all_files()
        self.disks = []
        self.overall_stats = {}
        self.record_size = args.record_size

    def verify_and_load_all_files(self):
        outdir_listing = [f for f in os.listdir(self.outdir) if os.path.isfile(os.path.join(self.outdir, f))]

        all_files = ['iostat.json', 'per_epoch_stats.json']

        load_and_proc_time_files = []
        
        for rank in range(self.comm_size):
            load_and_proc_time_files.append(f'{rank}_output.json')

        all_files.extend(load_and_proc_time_files)
        '''
        is_missing_file = False
        for necessary_file in all_files:
            if necessary_file not in outdir_listing:
                print(f"ERROR: missing necessary file: {os.path.join(self.outdir, necessary_file)}")
        if is_missing_file:
            exit(-1)
        '''
        with open(os.path.join(self.outdir, 'summary.json'), 'r') as summary_file:
            self.summary = json.load(summary_file)

        # All files are present, load some in
        try:
            with open(os.path.join(self.outdir, 'iostat.json'), 'r') as iotrace_file:
                self.iotrace = json.load(iotrace_file)
        except: 
            self.iotrace = None
            print(f"WARNING: missing necessary file: {os.path.join(self.outdir, 'iostat.json')}")

        try:
            with open(os.path.join(self.outdir, 'per_epoch_stats.json'), 'r') as per_epoch_stats_file:
                self.per_epoch_stats = json.load(per_epoch_stats_file)
        except: 
            self.per_epoch_stats = None
            print(f"WARNING: missing necessary file: {os.path.join(self.outdir, 'per_epoch_stats.json')}")

        # These ones will be loaded in later
        self.load_and_proc_time_files = [os.path.join(self.outdir, f) for f in load_and_proc_time_files]


    def process_loading_and_processing_times(self):

        logging.info(f"Calculating Loading and Processing Times")
        
        all_loading_times = []
        self.epoch_loading_times = {}
        
        all_processing_times = []
        self.epoch_processing_times = {}

        # Samples per second is straight forward, to obtain it 
        # we divide the batch size by the time taken to load it

        # Sample latency is defined by the time between when a sample is loaded
        # and when it is no longer needed. Since in a given epoch, we iterate over
        # batches once, a sample is no longer needed once the batch containing it 
        # has been processed. 
        # We obtain it by dividing the batch size by its processing time.
        all_sample_latencies = []
        all_sample_bandwidth = []
        self.epoch_sample_latencies = {}
        self.epoch_sample_bandwidth = {}
        self.num_files = len(self.load_and_proc_time_files)
        # There is one file per worker process, with data
        # separated by epoch and by phase of training (block, eval)
        # First, we will combine the different workers' data before
        # computing overall and per training phase statistics.
        for file in self.load_and_proc_time_files:
            logging.info(f"Reading from {file}")
            with open(file, 'r') as infile:
                load_and_proc_times = json.load(infile)

                for epoch in self.epochs_list:
                    logging.debug(f"Processing loading and processing times for epoch {epoch}")
                    loading_data = load_and_proc_times[epoch]['load']

                    if epoch not in self.epoch_loading_times:
                        # Initialize structures to hold the data
                        self.epoch_loading_times[epoch] = {}

                    for phase, phase_loading_times in loading_data.items():
                        assert isinstance(phase_loading_times, list)
                        logging.debug(f"Processing loading times for phase {phase}")

                        # The batch size might be different for training vs evals
                        if re.match(r'eval', phase):
                            effective_batch_size = self.batch_size_eval
                        else:
                            effective_batch_size = self.batch_size
                        
                        all_loading_times.extend(phase_loading_times)


                        if phase not in self.epoch_loading_times[epoch]:
                            self.epoch_loading_times[epoch][phase] = phase_loading_times
                        else:
                            self.epoch_loading_times[epoch][phase].extend(phase_loading_times)

                    # Same thing for processing times
                    processing_data = load_and_proc_times[epoch]['proc']

                    if epoch not in self.epoch_sample_latencies:
                        self.epoch_processing_times[epoch] = {}
                        self.epoch_sample_latencies[epoch] = {}
                        self.epoch_sample_bandwidth[epoch] = {}

                    # For each training phase, fetch the loading times and combine them
                    for phase, phase_processing_times in processing_data.items():
                        assert isinstance(phase_processing_times, list)
                        logging.debug(f"Processing processing times for phase {phase}")

                        # The batch size might be different for training vs evals
                        if re.match(r'eval', phase):
                            effective_batch_size = self.batch_size_eval
                        else:
                            effective_batch_size = self.batch_size
                        
                        all_processing_times.extend(phase_processing_times)

                        phase_sample_latencies = [effective_batch_size / time for time in phase_processing_times]
                        phase_sample_bandwidth = list(np.array(phase_sample_latencies)*self.record_size / 1024./1024)
                        all_sample_latencies.extend(phase_sample_latencies)
                        all_sample_bandwidth.extend(phase_sample_bandwidth)
                        if phase not in self.epoch_sample_latencies[epoch]:
                            self.epoch_processing_times[epoch][phase] = phase_processing_times
                            self.epoch_sample_latencies[epoch][phase] = phase_sample_latencies
                            self.epoch_sample_bandwidth[epoch][phase] = phase_sample_bandwidth 
                        else:
                            self.epoch_processing_times[epoch][phase].extend(phase_processing_times)
                            self.epoch_sample_latencies[epoch][phase].extend(phase_sample_latencies)
                            self.epoch_sample_bandwidth[epoch][phase].extend(phase_sample_bandwidth)



        # At this point, we should have one big structure containing overall stats, 
        # as well as all the combined loading and processing times for each phase of training
        
        logging.info(f"Computing overall stats")

        # Save the overall stats
        self.overall_stats['samples/s'] = self.get_stats(self.summary['metric']['train_throughput_samples_per_second'])
        io = np.array(self.summary['metric']['train_throughput_samples_per_second'])*self.record_size/1024/1024.
        self.overall_stats['MB/s'] = self.get_stats(io)
        # The average process loading time is the sum of all the time spent 
        # loading across different processes divided by the number of processes
        self.overall_stats['avg_process_loading_time'] = '{:.2f}'.format(sum(all_loading_times) / self.comm_size)
        # Same thing for average process processing time
        self.overall_stats['avg_process_processing_time'] = '{:.2f}'.format(sum(all_processing_times) / self.comm_size)

        logging.info(f"Computing per epoch stats")

        # Save the stats for each phase of training
        for epoch in self.epochs_list:

            epoch_loading_times = self.epoch_loading_times[epoch]
            epoch_processing_times = self.epoch_processing_times[epoch]
            epoch_sample_latencies = self.epoch_sample_latencies[epoch]
            epoch_sample_bandwidth = self.epoch_sample_bandwidth[epoch]
            for phase in epoch_loading_times.keys():
                logging.debug(f"Computing stats for epoch {epoch} {phase}")

                phase_loading_times = epoch_loading_times[phase]
                phase_processing_times = epoch_processing_times[phase]
                phase_sample_latencies = epoch_sample_latencies[phase]
                phase_sample_bandwidth = epoch_sample_bandwidth[phase]

                self.per_epoch_stats[epoch][phase]['avg_process_loading_time'] = '{:.2f}'.format(sum(phase_loading_times) / self.comm_size)
                self.per_epoch_stats[epoch][phase]['avg_process_processing_time'] = '{:.2f}'.format(sum(phase_processing_times) / self.comm_size)
                self.per_epoch_stats[epoch][phase]['samples/s'] = self.get_stats(phase_sample_latencies, num_procs=self.comm_size)
                self.per_epoch_stats[epoch][phase]['MB/s'] = self.get_stats(phase_sample_bandwidth, num_procs=self.comm_size)


    def get_stats(self, series, num_procs=1):
        """
        Return a dictionary with various statistics of the given series
        """

        if (num_procs>1):
            new_series = np.zeros(len(series)//num_procs)
            n = len(new_series)
            for i in range(num_procs):
                new_series += series[i*n:(i+1)*n]
            series = new_series
        if series is None or len(series) < 2:
            return {
                "mean": 'n/a',
                "std": 'n/a',
                "min": 'n/a',
                "median": 'n/a',
                "p90": 'n/a',
                "p99": 'n/a',
                "max": 'n/a'        
            }
        # Returns 99 cut points
        # We can use inclusive because we have the entire population
        percentiles = quantiles(series, n=100, method='inclusive')
        return {
            "mean": '{:.2f}'.format(mean(series)),
            "std": '{:.2f}'.format(stdev(series)),
            "min": '{:.2f}'.format(min(series)),
            "median": '{:.2f}'.format(median(series)),
            "p90": '{:.2f}'.format(percentiles[89]),
            "p99": '{:.2f}'.format(percentiles[98]),
            "max": '{:.2f}'.format(max(series))
        }


    def parse_iostat_trace(self):
        """
        Parse the iostat JSON file and return disk and cpu usage information
        """
        logging.info("Parsing iostat trace")
        # TODO: Support tracing on multiple hosts, here we only get data for the first
        iotrace = self.iotrace['sysstat']['hosts'][0]['statistics']
        # We will convert the iostat JSON output into a Dataframe indexed by timestamp 
        # Timestamps are already in UTC (when generated from within the container)
        # Pandas can read the format, then we can convert to numpy datetime64
        cpu_stats = pd.DataFrame(columns=['timestamp', 'user', 'system', 'iowait', 'steal', 'idle'])
        # The following columns are available:
        # ['timestamp', 'disk', 'r/s', 'w/s', 'rMB/s', 'wMB/s', 'r_await', 'w_await', 'rareq-sz', 'wareq-sz', 'aqu-sz'])
        disk_stats = pd.DataFrame(columns=['timestamp', 'disk', 'r/s', 'w/s', 'rMB/s', 'wMB/s', 'r_await', 'w_await', 'aqu-sz'])

        cpu_i = disk_i = 0
        for i, item in enumerate(iotrace):
            if i % 100 == 0:
                logging.info(f"Processing iostat item {i}")

            ts = item['timestamp']
            # Need to convert to UTC, this will depend on your timezone

            cpu = item['avg-cpu']
            # Combine user and nice cpu time into one for conciseness
            cpu_stats.loc[cpu_i] = [ts, cpu['user'] + cpu['nice'], cpu['system'], cpu['iowait'], cpu['steal'], cpu['idle']]
            cpu_i += 1
            # Add one row per disk
            for disk in item['disk']:
                row = [ts, disk['disk_device'], disk['r/s'], disk['w/s'], disk['rMB/s'], disk['wMB/s'], disk['r_await'], disk['w_await'], disk['aqu-sz']]
                disk_stats.loc[disk_i] = row
                disk_i += 1

        # Convert timestamp fields to datatime
        cpu_stats.timestamp = pd.to_datetime(cpu_stats.timestamp)
        disk_stats.timestamp = pd.to_datetime(disk_stats.timestamp)
        self.disk_stats = disk_stats
        self.disks = pd.unique(self.disk_stats['disk'])
        self.cpu_stats = cpu_stats


    def extract_stats_from_iostat_trace(self):
        logging.info("Extracting stats from iostat trace")

        # Helper functions
        def get_series_daterange(series, start, end): 
            data = series[series['timestamp'] >= start]
            data = data[data['timestamp'] < end]
            return data

        def addto_and_return_stats(addto, df, stat):
            data = df[stat].to_list()
            addto += data
            if len(data) < 2:
                logging.warning(f'Less than 2 data points for {stat}')
            return self.get_stats(data)
        
        r_overall_bandwidth = {}
        w_overall_bandwidth = {}
        r_overall_iops = {}
        w_overall_iops = {}
        r_overall_wait = {}
        w_overall_wait = {}
        overall_aqu_sz = {}

        cpu_overall_user = []
        cpu_overall_sys = []
        cpu_overall_iowait = []
        cpu_overall_steal = []
        cpu_overall_idle = []

        disk_stats_to_extract = ['rMB/s', 'wMB/s', 'r/s', 'w/s', 'r_await', 'w_await', 'aqu-sz']
        disk_accumulators = [r_overall_bandwidth, w_overall_bandwidth, r_overall_iops, w_overall_iops, r_overall_wait, w_overall_wait, overall_aqu_sz]
        cpu_stats_to_extract = ['user', 'system', 'iowait', 'steal', 'idle']
        cpu_accumulators = [cpu_overall_user, cpu_overall_sys, cpu_overall_iowait, cpu_overall_steal, cpu_overall_idle]

        # Initialize disk accumulators
        for disk in self.disks:
            for acc in disk_accumulators:
                acc[disk] = []

        for epoch in self.epochs_list:


            epoch_data = self.per_epoch_stats[epoch]

            for phase, phase_data in epoch_data.items():
                logging.info(f"Extracting stats for epoch {epoch} {phase}")

                if not isinstance(phase_data, dict):
                    continue

                start, end = pd.to_datetime(phase_data['start']), pd.to_datetime(phase_data['end'])

                disk_io = get_series_daterange(self.disk_stats, start, end)

                self.per_epoch_stats[epoch][phase]['disk'] = {}

                for disk in self.disks:

                    self.per_epoch_stats[epoch][phase]['disk'][disk] = {}

                    disk_data = disk_io[disk_io['disk'] == disk]

                    for i, stat in enumerate(disk_stats_to_extract):
                        data = disk_data[stat].to_list()
                        disk_accumulators[i][disk] += data
                        self.per_epoch_stats[epoch][phase]['disk'][disk][stat] = addto_and_return_stats(disk_accumulators[i][disk], disk_data, stat)

                cpu_data = get_series_daterange(self.cpu_stats, start, end)

                self.per_epoch_stats[epoch][phase]['cpu'] = {}
                for i, stat in enumerate(cpu_stats_to_extract):
                    self.per_epoch_stats[epoch][phase]['cpu'][stat] = addto_and_return_stats(cpu_accumulators[i], cpu_data, stat)


        # Compute overall stats for each disk
        self.overall_stats['disk'] = {}
        for disk in self.disks:
            self.overall_stats['disk'][disk] = {}
            self.overall_stats['disk'][disk]['rMB/s'] = self.get_stats(r_overall_bandwidth[disk])
            self.overall_stats['disk'][disk]['wMB/s'] = self.get_stats(w_overall_bandwidth[disk])
            self.overall_stats['disk'][disk]['r/s'] = self.get_stats(r_overall_iops[disk])
            self.overall_stats['disk'][disk]['w/s'] = self.get_stats(w_overall_iops[disk])
            self.overall_stats['disk'][disk]['r_await'] = self.get_stats(r_overall_wait[disk])
            self.overall_stats['disk'][disk]['w_await'] = self.get_stats(w_overall_wait[disk])
            self.overall_stats['disk'][disk]['aqu-sz'] = self.get_stats(overall_aqu_sz[disk])

        self.overall_stats['cpu'] = {
            'user': self.get_stats(cpu_overall_user),
            'system': self.get_stats(cpu_overall_sys),
            'iowait': self.get_stats(cpu_overall_iowait),
            'steal': self.get_stats(cpu_overall_steal),
            'idle': self.get_stats(cpu_overall_idle)
        }

    def write_report(self):
        logging.info("Writing report")

        TAB = ' ' * 4
        HALF_TAB = ' ' * 2
        TABLE_HEADER = ['mean', 'std', 'min', 'median', 'p90', 'p99', 'max']
        ROW_SEP = "------------------------------------------------------------------------------------------"

        # Helper methods for formatting
        def format_list(l):
            format = "{:>12} " * len(l)
            return format.format(*l)
                
        def format_stats(stats):
            if isinstance(stats, dict):
                format = "{:>12} " * len(stats.keys())
                stats = format.format(*stats.values())
            return stats

        def format_print(outfile, content, indent=0):
            indent = " " * 4 * indent
            max_row_name_len = 0
            for k in content.keys():
                if len(k) > max_row_name_len:
                    max_row_name_len = len(k)

            left_align_space = max_row_name_len + 8
            fmt = "{:<" + f'{left_align_space}' + "}"

            for row_name, row_content in content.items():
                outfile.write(f"{indent}{fmt.format(row_name)}{row_content}\n")
            outfile.write("\n")

        def write_out_stats_table(outfile, stats_dict, has_loading=True, indent=0, overall=False):
            if self.iotrace == None:
                return 
            indent = TAB * indent

            # This value should be large enough to hold the largest field name + all inner tab-ing + a margin
            left_align_space = len("W Bandwidth (MB/s):") + len(TAB) + len(HALF_TAB) + 10
            fmt = "{:<" + f'{left_align_space}' + "}"

            outfile.write(f"{indent}{fmt.format('')}{format_list(TABLE_HEADER)}\n")
            outfile.write(f"{indent}{fmt.format('')}{ROW_SEP}\n")

            if has_loading:
                if overall:
                    outfile.write(f"{indent}{fmt.format('Throughput Stats (over all epochs)')}\n")
                    outfile.write(f"{indent}{fmt.format('  Samples/s:')}{format_stats(stats_dict['samples/s'])}\n")
                    outfile.write(f"{indent}{fmt.format('  MB/s (derived from Samples/s):')}{format_stats(stats_dict['MB/s'])}\n")
                else:
                    outfile.write(f"{indent}{fmt.format('Throughput Stats (over all steps)')}\n")
                    outfile.write(f"{indent}{fmt.format('  Samples/s:')}{format_stats(stats_dict['samples/s'])}\n")
                    outfile.write(f"{indent}{fmt.format('  MB/s (derived from Samples/s):')}{format_stats(stats_dict['MB/s'])}\n")

            outfile.write("\n")
            outfile.write(f"{indent}{fmt.format('I/O Stats (over all time segments)')}\n")

            for disk in self.disks:
                outfile.write(f"{indent}{fmt.format(f'{HALF_TAB}Device: {disk}')}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}R Bandwidth (MB/s):')}{format_stats(stats_dict['disk'][disk]['rMB/s'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}W Bandwidth (MB/s):')}{format_stats(stats_dict['disk'][disk]['wMB/s'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}R IOPS:')}{format_stats(stats_dict['disk'][disk]['r/s'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}W IOPS:')}{format_stats(stats_dict['disk'][disk]['w/s'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}Avg R Time (ms):')}{format_stats(stats_dict['disk'][disk]['r_await'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}Avg W Time (ms):')}{format_stats(stats_dict['disk'][disk]['w_await'])}\n")
                outfile.write(f"{indent}{fmt.format(f'{TAB}Avg Queue Length:')}{format_stats(stats_dict['disk'][disk]['aqu-sz'])}\n\n")

            outfile.write(f"{indent}{fmt.format('CPU Stats')}\n")

            outfile.write(f"{indent}{fmt.format(f'{TAB}User (%):')}{format_stats(stats_dict['cpu']['user'])}\n")
            outfile.write(f"{indent}{fmt.format(f'{TAB}System (%):')}{format_stats(stats_dict['cpu']['system'])}\n")
            outfile.write(f"{indent}{fmt.format(f'{TAB}IO Wait (%):')}{format_stats(stats_dict['cpu']['iowait'])}\n")
            outfile.write(f"{indent}{fmt.format(f'{TAB}Steal (%):')}{format_stats(stats_dict['cpu']['steal'])}\n")
            outfile.write(f"{indent}{fmt.format(f'{TAB}Idle (%):')}{format_stats(stats_dict['cpu']['idle'])}\n")
            outfile.write("\n")

        # Get overall start, end and duration of the run
        self.overall_stats['start'] = pd.to_datetime(self.per_epoch_stats["1"]['start'])
        self.overall_stats['end'] = pd.to_datetime(self.per_epoch_stats[str(self.epochs)]['end'])
        duration = self.overall_stats['end'] - self.overall_stats['start'] 
        self.overall_stats['duration'] = '{:.2f}'.format(duration.total_seconds())

        if self.name != "":
            report_name = f'DLIO_{self.name}_report.txt'
        else:
            report_name = 'DLIO_report.txt'

        # Write the report
        with open(os.path.join(self.outdir, report_name), 'w') as outfile:

            outfile.write("DLIO v1.0 Report\n\n")
            outfile.write("Note: Training phases lasting less than 2 seconds, will show 'n/a' values, as there is not enough data to compute statistics.\n\n")
            outfile.write("Overall\n\n")
            
            overall_desc = {
                'Run name:': self.name,
                'Started:': self.overall_stats['start'],
                'Ended:': self.overall_stats['end'],
                'Duration (s):': self.overall_stats['duration'],
                'Num Ranks:': self.comm_size,
                'Batch size (per rank):': self.batch_size,
            }

            if self.do_eval:
                overall_desc['Eval batch size:'] = self.batch_size_eval

            format_print(outfile, overall_desc, indent=1)
            if (self.iotrace is not None):
                write_out_stats_table(outfile, self.overall_stats, indent=1, overall=True)

            outfile.write("\nDetailed Report\n\n")

            i_blk = i_eval = i_ckpt = 1
            for epoch in self.epochs_list:
                epoch_data = self.per_epoch_stats[epoch]
                
                outfile.write(f"Epoch {epoch}\n")

                epoch_desc = {
                    'Started:': pd.to_datetime(epoch_data['start']),
                    'Ended:': pd.to_datetime(epoch_data['end']),
                    'Duration (s):': epoch_data['duration']
                }
                format_print(outfile, epoch_desc, indent=1)

                for phase, phase_data in epoch_data.items():
                    # Skip fields like epoch start, end, duration
                    if not isinstance(phase_data, dict):
                        continue
                    
                    has_loading = True
                    if re.match(r'block\d+', phase):
                        outfile.write(f"{TAB}Block {i_blk}\n")
                        i_blk += 1
                    elif re.match(r'eval\d*', phase):
                        outfile.write(f"{TAB}Eval {i_eval}\n")
                        i_eval += 1
                    elif re.match(r'ckpt\d+', phase):
                        outfile.write(f"{TAB}Checkpoint {i_ckpt}\n")
                        has_loading = False
                        i_ckpt += 1
                    else:
                        print("Warning: unknown training phase")
                        outfile.write(f"{TAB}{phase}\n")

                    phase_desc = {
                        'Started:': pd.to_datetime(phase_data['start']),
                        'Ended:': pd.to_datetime(phase_data['end']),
                        'Duration (s):': phase_data['duration'],
                    }

                    if has_loading:
                        phase_desc['Avg loading time / rank (s):'] = phase_data['avg_process_loading_time']
                        phase_desc['Avg processing time / rank (s):'] = phase_data['avg_process_processing_time']

                    format_print(outfile, phase_desc, indent=2)
                    write_out_stats_table(outfile, phase_data, has_loading=has_loading, indent=2)

        logging.info(f"Successfully wrote {os.path.join(self.outdir, report_name)}")


    def generate_report(self):
        logging.info(f"Generating Report")
        self.process_loading_and_processing_times()
        # parse iostat report
        if self.iotrace is not None: 
            self.parse_iostat_trace()
            self.extract_stats_from_iostat_trace()
        # Write the report
        self.write_report()
import yaml
from yaml.loader import SafeLoader



def main():
    """
    The main method to start the benchmark runtime.
    """
    parser = argparse.ArgumentParser(description='DLIO PostProcessor')

    parser.add_argument("-of", "--output-folder", default="./output", type=str,
                        help="Folder containing the output of a benchmark run.")
    parser.add_argument("-hf", "--hydra-folder", default="./.hydra", type=str,
                        help="Hydra folder containing configs")
    parser.add_argument("-np", "--num-proc", default=1, type=int,
                        help="Number of processes that were ran.")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help="Number of epochs to be emulated within benchmark.")
    parser.add_argument("-bs", "--batch-size", default=1, type=int,
                        help="Per worker batch size for training records.")
    parser.add_argument("-de", "--do-eval", default=False, type=str2bool,
                        help="If evaluations were simulated.")
    parser.add_argument("-bse", "--batch-size-eval", default=1, type=int,
                        help="Per worker batch size for evaluation records.")
    parser.add_argument("-c", "--do-checkpoint", default=False, type=str2bool,
                        help="If checkpointing was simulated")
    parser.add_argument("-d", "--debug", default=False, type=str2bool,
                        help="Print out more logging")
    parser.add_argument("-n", "--name", default="", type=str,
                        help="Name of the run")
    orig_args = parser.parse_args()
    args = parser.parse_args()

    # figuring out the number of process from the outputs
    args.num_proc = len(glob.glob(args.output_folder + "/*_output.json"))

    # load the yaml file and override the command line argument
    base_config = os.path.join(args.output_folder, args.hydra_folder, "config.yaml")
    override_config = os.path.join(args.output_folder, args.hydra_folder, "overrides.yaml")
    with open(base_config) as f:
        hydra_config  = yaml.load(f, Loader=SafeLoader)
    LoadConfig(args, hydra_config['workload'])
    if 'model' in hydra_config['workload']:
        args.name = hydra_config['workload']['model']['name']
    else:
        args.name="default"
    args.record_size = hydra_config['workload']['dataset']['record_length']
    for op in open(override_config, "r").readlines():
        if op.find("train.epochs")!=-1:
            args.epochs = int(op.split("=")[1])
        if op.find('batch_size=')!=-1:
            args.batch_size = int(op.split("=")[1])
        if op.find("batch_size_eval")!=-1:
            args.batch_size_eval = int(op.split("=")[1])
        if op.find('workflow.checkpoint')!=-1:
            args.do_checkpoint=str2bool(op.split("=")[1])
        if op.find("debug")!=-1:
            args.debug = str2bool(op.split("=")[1])

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S")

    print(f"===============Processing DLIO output================")
    print(f"  Job configuration")

    for arg in vars(orig_args):
        print(f"  {arg}: {getattr(args, arg)}")
    postproc = DLIOPostProcessor(args)
    postproc.generate_report()

if __name__ == '__main__':
    main()
    exit(0)
