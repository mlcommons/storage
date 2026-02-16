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
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, DLIOLogger

import os
import json
import math
import pandas as pd
from time import time
import numpy as np
import psutil
import platform
import socket
from mpi4py import MPI
def lines_to_dict(lines):
    dict = {}
    for l in lines.split("\n"):
        if len(l.split(":"))==2: 
            k, v = l.split(":")
            if k[-1] == "\n":
                k = k[:-1]
            k = k.strip()
            v = v.strip()
        if k != 'processor':
            dict[k] = v
    return dict

class StatsCounter(object):

    def __init__(self):
        self.MPI = DLIOMPI.get_instance()
        self.logger = DLIOLogger.get_instance()
        self.comm = self.MPI.comm()
        self.args = ConfigArguments.get_instance()
        self.my_rank = self.args.my_rank
        self.comm_size = self.args.comm_size
        self.output_folder = self.args.output_folder
        self.record_size = self.args.record_length
        self.batch_size = self.args.batch_size
        self.batch_size_eval = self.args.batch_size_eval
        self.checkpoint_size = 0.0
        self.summary = {}
        self.summary['start'] = utcnow()
        self.summary['num_accelerators'] = self.comm_size
        self.summary['num_hosts'] = self.MPI.nnodes()
        self.summary['hostname'] = socket.gethostname()
        self.summary['metric'] = {}
        self.summary['num_files_train'] = self.args.num_files_train
        self.summary['num_files_eval'] = self.args.num_files_eval
        self.summary['num_samples_per_file'] = self.args.num_samples_per_file
        self.summary['host_cpu_count'] = psutil.cpu_count()
        self.summary['host_processor_name'] = platform.processor()
        self.summary['potential_caching'] = False

        if os.path.exists("/proc/cpuinfo"):
            self.summary['host_cpuinfo'] = lines_to_dict(open("/proc/cpuinfo", "r").read())
        if os.path.exists("/proc/meminfo"):
            self.summary['host_meminfo'] = lines_to_dict(open("/proc/meminfo", "r").read())
        max_steps = math.floor(self.args.num_samples_per_file * self.args.num_files_train / self.args.batch_size / self.args.comm_size)

        if self.args.total_training_steps > 0:
            if self.args.total_training_steps > max_steps:
                self.logger.error(f"Only have enough data for {max_steps} steps but {self.args.total_training_steps} wanted")
                exit(-1)
            self.steps_override = True
            self.steps = self.args.total_training_steps
        else:
            self.steps_override = False
            self.steps = max_steps
        self.metric_steps = self.steps - (self.args.metric_exclude_end_steps + self.args.metric_exclude_start_steps)
        self.metric_start_step = self.args.metric_exclude_start_steps
        self.metric_end_step = self.steps - 1 - self.args.metric_exclude_end_steps 
        if self.comm.rank == 0:
            self.logger.info(f"{utcnow()} Metric calculation will exclude the beginning {self.args.metric_exclude_start_steps} and end {self.args.metric_exclude_end_steps} steps, only includes {self.metric_steps} steps.")
        self.steps_eval = math.floor(self.args.num_samples_per_file * self.args.num_files_eval / self.args.batch_size_eval / self.args.comm_size)
        self.per_epoch_stats = {}
        self.metric_steps_eval = self.steps_eval - (self.args.metric_exclude_end_steps + self.args.metric_exclude_start_steps)
        self.metric_start_step_eval = self.args.metric_exclude_start_steps
        self.metric_end_step_eval = self.steps_eval - 1 - self.args.metric_exclude_end_steps 
        # Only the root process keeps track of overall stats
        # Each process keeps track of its loading and processing times independently
        self.output = {}
        self.output['host_memory_GB'] = psutil.virtual_memory().total/1024./1024./1024
        host_memory = np.zeros(self.MPI.nnodes())
        host_memory_agg = np.zeros(self.MPI.nnodes())
        if self.MPI.local_rank()==0:
            host_memory[self.MPI.node()] = self.output['host_memory_GB']
        self.MPI.comm().Reduce(host_memory, host_memory_agg, op=MPI.SUM, root=0)
        self.summary['host_memory_GB'] = list(host_memory_agg)
        self.output['host_cpu_count'] = psutil.cpu_count()
        cpu_count = np.zeros(self.MPI.nnodes())
        cpu_count_agg = np.zeros(self.MPI.nnodes())
        if self.MPI.local_rank()==0:
            cpu_count[self.MPI.node()] = self.output['host_cpu_count']
        self.MPI.comm().Reduce(cpu_count, cpu_count_agg, op=MPI.SUM, root=0)   

        self.summary['host_cpu_count'] = [int(d) for d in cpu_count_agg]
        self.output['host_processor_name'] = platform.processor()
        self.output['potential_caching'] = 0
        if os.path.exists("/proc/cpuinfo"):
            self.output['host_cpuinfo'] = lines_to_dict(open("/proc/cpuinfo", "r").read())
        if os.path.exists("/proc/meminfo"):
            self.output['host_meminfo'] = lines_to_dict(open("/proc/meminfo", "r").read())

        self.train_au = []
        self.eval_au = []
        self.train_throughput = []
        self.eval_throughput = []
        data_per_node = self.MPI.npernode()*self.args.num_samples_per_file * self.args.num_files_train//self.MPI.size()*self.args.record_length
        self.summary['data_size_per_host_GB'] = data_per_node/1024./1024./1024.
        if self.MPI.rank() == 0 and self.args.do_train:
            self.logger.info(f"Total amount of data each host will consume is {data_per_node/1024./1024./1024} GB; each host has {self.summary['host_memory_GB']} GB memory") 
        if self.summary['data_size_per_host_GB'] <= self.output['host_memory_GB']:
            self.output['potential_caching'] = 1
            if self.MPI.rank() == 0 and self.args.do_train: 
                self.logger.warning("The amount of dataset is smaller than the host memory; data might be cached after the first epoch. Increase the size of dataset to eliminate the caching effect!!!")
        potential_caching = []
        for i in range(self.MPI.nnodes()):
            if self.summary['host_memory_GB'][i]  <= self.summary['data_size_per_host_GB']:
                potential_caching.append(0)
            else:
                potential_caching.append(1)
        self.summary['potential_caching'] = potential_caching

    def start_run(self):
        self.start_run_timestamp = time()
    def end_run(self):
        self.end_run_timestamp = time()
        if self.args.do_checkpoint and self.my_rank == 0:
            duration_save = []
            io_save = []
            duration_load = []
            io_load = []
            for e in self.per_epoch_stats:
                for t in self.per_epoch_stats[e]:
                    if t.find("save_ckpt")!=-1:
                        duration_save.append(float(self.per_epoch_stats[e][t]['duration']))
                        io_save.append(self.per_epoch_stats[e][t]['throughput'])
                    elif t.find("load_ckpt")!=-1:
                        duration_load.append(float(self.per_epoch_stats[e][t]['duration']))
                        io_load.append(self.per_epoch_stats[e][t]['throughput'])
            self.summary['metric']['save_checkpoint_io_mean_GB_per_second'] = np.mean(io_save)
            self.summary['metric']['save_checkpoint_io_stdev_GB_per_second'] = np.std(io_save)
            self.summary['metric']['save_checkpoint_duration_mean_seconds'] = np.mean(duration_save)
            self.summary['metric']['save_checkpoint_duration_stdev_seconds'] = np.std(duration_save)
            if len(io_load) > 0:
                self.summary['metric']['load_checkpoint_io_mean_GB_per_second'] = np.mean(io_load)
                self.summary['metric']['load_checkpoint_io_stdev_GB_per_second'] = np.std(io_load)
                self.summary['metric']['load_checkpoint_duration_mean_seconds'] = np.mean(duration_load)
                self.summary['metric']['load_checkpoint_duration_stdev_seconds'] = np.std(duration_load)
            self.summary['metric']['checkpoint_size_GB'] = self.checkpoint_size
        if not self.args.generate_only:
            total_elapsed_time = self.end_run_timestamp - self.start_run_timestamp
            train_au = np.array(self.comm.allreduce(np.array(self.train_au)))/self.comm.size
            train_throughput = self.comm.allreduce(np.array(self.train_throughput))
            self.summary['epochs'] = len(train_au)
            if self.args.do_train:
                self.summary['metric']['train_au_percentage'] = list(train_au)
                self.summary['metric']['train_au_mean_percentage'] = np.mean(train_au)
                if self.summary['metric']['train_au_mean_percentage'] >=self.args.au*100:
                    self.summary['metric']['train_au_meet_expectation'] = 'success'
                else:
                    self.summary['metric']['train_au_meet_expectation'] = 'fail'
                self.summary['metric']['train_au_stdev_percentage'] = np.std(train_au)
                self.summary['metric']['train_throughput_samples_per_second'] = list(train_throughput)
                self.summary['metric']['train_throughput_mean_samples_per_second'] = np.mean(train_throughput)
                self.summary['metric']['train_throughput_stdev_samples_per_second'] = np.std(train_throughput)
                self.summary['metric']['train_io_mean_MB_per_second'] = np.mean(train_throughput)*self.record_size/1024./1024.
                self.summary['metric']['train_io_stdev_MB_per_second'] = np.std(train_throughput)*self.record_size/1024./1024.
            
            if self.args.do_eval:
                eval_au = np.array(self.comm.allreduce(self.eval_au))/self.comm.size
                eval_throughput = self.comm.allreduce(self.eval_throughput)
                self.summary['metric']['eval_au_percentage'] = list(eval_au)
                self.summary['metric']['eval_au_mean_percentage'] = np.mean(eval_au)
                if self.summary['metric']['eval_au_mean_percentage'] >=self.args.au*100:
                    self.summary['metric']['eval_au_meet_expectation'] = 'success'
                else:
                    self.summary['metric']['eval_au_meet_expectation'] = 'fail'
                self.summary['metric']['eval_au_stdev_percentage'] = np.std(eval_au)
                self.summary['metric']['eval_throughput_samples_per_second'] = list(eval_throughput)
                self.summary['metric']['eval_throughput_mean_samples_per_second'] = np.mean(eval_throughput)
                self.summary['metric']['eval_throughput_stdev_samples_per_second'] = np.std(eval_throughput)
                self.summary['metric']['eval_io_mean_MB_per_second'] = np.mean(eval_throughput)*self.record_size/1024./1024.
                self.summary['metric']['eval_io_stdev_MB_per_second'] = np.std(eval_throughput)*self.record_size/1024./1024.
            if self.my_rank==0:
                self.logger.output(f"{utcnow()} Saved outputs in {self.output_folder}")   
                metric="Averaged metric over all steps/epochs\n[METRIC] ==========================================================\n"
                metric = metric + f"[METRIC] Number of Simulated Accelerators: {self.comm_size} \n"
                if self.args.do_train:
                    metric = metric + f"[METRIC] Training Accelerator Utilization [AU] (%): {np.mean(train_au):.4f} ({np.std(train_au):.4f})\n"
                    metric = metric + f"[METRIC] Training Throughput (samples/second): {np.mean(train_throughput):.4f} ({np.std(train_throughput):.4f})\n"
                    metric = metric + f"[METRIC] Training I/O Throughput (MB/second): {np.mean(train_throughput)*self.record_size/1024/1024:.4f} ({np.std(train_throughput)*self.record_size/1024/1024:.4f})\n"
                    metric = metric + f"[METRIC] train_au_meet_expectation: {self.summary['metric']['train_au_meet_expectation']}\n"
                if self.args.do_checkpoint: 
                    if self.args.num_checkpoints_write > 0:
                        metric = metric + f"[METRIC] Checkpoint save duration (seconds): {self.summary['metric']['save_checkpoint_duration_mean_seconds']:.4f} ({self.summary['metric']['save_checkpoint_duration_stdev_seconds']:.4f})\n"
                        metric = metric + f"[METRIC] Checkpoint save I/O Throughput (GB/second): {self.summary['metric']['save_checkpoint_io_mean_GB_per_second']:.4f} ({self.summary['metric']['save_checkpoint_io_stdev_GB_per_second']:.4f})\n"
                    if self.args.num_checkpoints_read > 0:
                        metric = metric + f"[METRIC] Checkpoint load duration (seconds): {self.summary['metric']['load_checkpoint_duration_mean_seconds']:.4f} ({self.summary['metric']['load_checkpoint_duration_stdev_seconds']:.4f})\n"
                        metric = metric + f"[METRIC] Checkpoint load I/O Throughput (GB/second): {self.summary['metric']['load_checkpoint_io_mean_GB_per_second']:.4f} ({self.summary['metric']['load_checkpoint_io_stdev_GB_per_second']:.4f})\n"

                if self.args.do_eval:
                    metric = metric + f"[METRIC] Eval Accelerator Utilization [AU] (%): {np.mean(eval_au):.4f} ({np.std(eval_au):.4f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (samples/second): {np.mean(eval_throughput):.6f} ({np.std(eval_throughput):.6f})\n"
                    metric = metric + f"[METRIC] Eval Throughput (MB/second): {np.mean(eval_throughput)*self.record_size/1024/1024:.6f} ({np.std(eval_throughput)*self.record_size/1024/1024:.6f})\n"
                    metric = metric + f"[METRIC] eval_au_meet_expectation: {self.summary['metric']['eval_au_meet_expectation']}\n"
                metric+="[METRIC] ==========================================================\n"
                self.logger.output(metric)   
    def start_train(self, epoch):   
        ts = utcnow()
        self.per_epoch_stats[epoch] = {
            'start': ts,
        }
        if self.my_rank == 0:
            if self.steps_override:
                self.logger.output(f"{ts} Starting epoch {epoch}: Overriding number of steps to {self.steps}.")
            else:
                self.logger.output(f"{ts} Starting epoch {epoch}: {self.steps} steps expected")
        # Initialize dicts for the current epoch
        self.output[epoch] = {}
        self.output[epoch]['load'] = {}
        self.output[epoch]['proc'] = {}
        self.output[epoch]['throughput'] = {}
        self.output[epoch]['au'] = {}
        self.output[epoch]['compute'] = {}
        if os.path.exists("/proc/meminfo"):
            self.output[epoch]['host_meminfo'] = lines_to_dict(open("/proc/meminfo", "r").read())

    def end_train(self, epoch, steps):
        au = np.array([self.output[epoch]['au'][k] for k in self.output[epoch]['au']])
        throughput = np.array([self.output[epoch]['throughput'][k] for k in self.output[epoch]['throughput']])
        steps = np.array([len(self.output[epoch]['proc'][k]) for k in self.output[epoch]['throughput']])
        if (np.sum(steps)==0):
            au = 0.0
            throughput = 0.0
        else:
            au = np.sum(au*steps)/np.sum(steps)
            throughput = np.sum(throughput*steps)/np.sum(steps)
        self.train_au.append(au)
        self.train_throughput.append(throughput)

        ts = utcnow()
        duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch]['start'])
        duration = '{:.2f}'.format(duration.total_seconds())
        self.per_epoch_stats[epoch]['end'] = ts
        self.per_epoch_stats[epoch]['duration'] = duration
        if self.my_rank == 0:
            self.logger.output(f"{ts} Ending epoch {epoch} - {np.sum(steps)} steps completed in {duration} s")

    def start_eval(self, epoch):
        self.start_timestamp = time()
        ts = utcnow()
        self.per_epoch_stats[epoch]['eval'] = {
            'start': ts
        }
        if self.my_rank == 0:
            self.logger.output(f"{ts} Starting eval - {self.steps_eval} steps expected")
        self.output[epoch]['load']['eval'] = []
        self.output[epoch]['proc']['eval'] = []
        self.output[epoch]['compute']['eval'] = []
        self.output[epoch]['au']['eval'] = 0.0
        self.output[epoch]['throughput']['eval'] = 0.0
    def end_eval(self, epoch):
        self.end_timestamp = time()
        self.compute_metrics_eval(epoch)
        self.eval_au.append(self.output[epoch]['au']['eval'])
        self.eval_throughput.append(self.output[epoch]['throughput']['eval'] )
        ts = utcnow()
        duration = pd.to_datetime(ts)- pd.to_datetime(self.per_epoch_stats[epoch]['eval']['start'])
        duration = '{:.2f}'.format(duration.total_seconds())
        self.per_epoch_stats[epoch]['eval']['end'] = ts
        self.per_epoch_stats[epoch]['eval']['duration'] = duration  
        if self.my_rank == 0:
            self.logger.output(f"{ts} Ending eval - {self.steps_eval} steps completed in {duration} s")
            self.logger.output(f"{utcnow()} Epoch {epoch} [Eval] Accelerator Utilization [AU] (%): {self.output[epoch]['au']['eval']:.4f}")
            self.logger.output(f"{utcnow()} Epoch {epoch} [Eval] Throughput (samples/second): {self.output[epoch]['throughput']['eval']*self.comm_size:.4f}")

    def start_epoch(self, epoch=1):
        ts = utcnow()
        if not(epoch in self.output):
            self.output[epoch] = {'start': ts}
            self.output[epoch]['load'] = {}
            self.output[epoch]['proc'] = {}
            self.output[epoch]['throughput'] = {}
            self.output[epoch]['au'] = {}
            self.output[epoch]['compute'] = {}
        if not(epoch in self.per_epoch_stats):
            self.per_epoch_stats[epoch] = {'start': ts}
    def end_epoch(self, epoch=1):
        ts = utcnow()
        self.output[epoch]['end'] = ts
        self.per_epoch_stats[epoch]['end']=ts

    def start_block(self, epoch, block):
        self.start_timestamp = time()
        self.output[epoch]['load'][f'block{block}'] = []
        self.output[epoch]['proc'][f'block{block}'] = []
        self.output[epoch]['throughput'][f'block{block}'] = 0.0
        self.output[epoch]['au'][f'block{block}'] = 0.0
        self.output[epoch]['compute'][f'block{block}'] = []
        ts = utcnow()
        self.per_epoch_stats[epoch][f'block{block}'] = {
            'start': ts
        }
        if self.my_rank == 0:
            self.logger.output(f"{ts} Starting block {block}")

    def end_block(self, epoch, block, steps_taken):
        self.end_timestamp = time()
        self.compute_metrics_train(epoch, block)
        if 'end' in self.per_epoch_stats[epoch][f'block{block}']:
            return
        ts = utcnow()
        duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'block{block}']['start'])
        duration = '{:.2f}'.format(duration.total_seconds())
        self.per_epoch_stats[epoch][f'block{block}']['end'] = ts
        self.per_epoch_stats[epoch][f'block{block}']['duration'] = duration

        if self.my_rank == 0:
            self.logger.output(f"{ts} Ending block {block} - {steps_taken} steps completed in {duration} s")
            if self.args.do_train:
                self.logger.output(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Accelerator Utilization [AU] (%): {self.output[epoch]['au'][f'block{block}']:.4f}")
                self.logger.output(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Throughput (samples/second): {self.output[epoch]['throughput'][f'block{block}']*self.comm_size:.4f}")
                self.logger.output(f"{utcnow()} Epoch {epoch} - Block {block} [Training] Computation time per step (second): {np.mean(self.output[epoch]['compute'][f'block{block}'][self.metric_start_step:self.metric_end_step+1]):.4f}+/-{np.std(self.output[epoch]['compute'][f'block{block}'][self.metric_start_step:self.metric_end_step+1]):.4f} (set value: {self.args.computation_time})")

    def start_save_ckpt(self, epoch, block, steps_taken):
        ts = utcnow()
        if self.my_rank == 0:
            self.logger.output(f"{ts} Starting saving checkpoint {block} after total step {steps_taken} for epoch {epoch}")
        self.per_epoch_stats[epoch][f'save_ckpt{block}'] = {
                'start': ts
        }

    def end_save_ckpt(self, epoch, block):
        ts = utcnow()
        duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'save_ckpt{block}']['start'])
        self.per_epoch_stats[epoch][f'save_ckpt{block}']['end'] = ts
        self.per_epoch_stats[epoch][f'save_ckpt{block}']['duration'] = float(duration.total_seconds())
        self.per_epoch_stats[epoch][f'save_ckpt{block}']['throughput'] = self.checkpoint_size / float(duration.total_seconds())
        if self.my_rank == 0:
            self.logger.output(f"{ts} Finished saving checkpoint {block} for epoch {epoch} in {duration.total_seconds():.4f} s; Throughput: {self.per_epoch_stats[epoch][f'save_ckpt{block}']['throughput']:.4f} GB/s")

    def start_load_ckpt(self, epoch, block, steps_taken):
        ts = utcnow()
        if self.my_rank == 0:
             self.logger.output(f"{ts} Starting loading checkpoint {block} after total step {steps_taken} for epoch {epoch}")
        self.per_epoch_stats[epoch][f'load_ckpt{block}'] = {
                'start': ts
        }
      
    def end_load_ckpt(self, epoch, block):
        ts = utcnow()
        duration = pd.to_datetime(ts) - pd.to_datetime(self.per_epoch_stats[epoch][f'load_ckpt{block}']['start'])
        self.per_epoch_stats[epoch][f'load_ckpt{block}']['end'] = ts
        self.per_epoch_stats[epoch][f'load_ckpt{block}']['duration'] = float(duration.total_seconds())
        self.per_epoch_stats[epoch][f'load_ckpt{block}']['throughput'] = self.checkpoint_size / float(duration.total_seconds())
        if self.my_rank == 0:
            self.logger.output(f"{ts} Finished loading checkpoint {block} for epoch {epoch} in {duration.total_seconds():.4f} s; Throughput: {self.per_epoch_stats[epoch][f'load_ckpt{block}']['throughput']:.4f} GB/s")

    def start_loading(self):
        self.start_time_loading = time()
    def start_compute(self):
        self.start_time_compute = time()
    def batch_loaded(self, epoch, step, block):
        duration = time() - self.start_time_loading
        key = f'block{block}'
        if key in self.output[epoch]['load']:
            self.output[epoch]['load'][key].append(duration)
        else:
            self.output[epoch]['load'][key] = [duration]
        self.logger.info(f"{utcnow()} Rank {self.my_rank} step {step}: loaded {self.batch_size} samples in {duration:.4f} s")

    def batch_processed(self, epoch, step, block):
        current_time = time()
        duration = current_time - self.start_time_loading 
        key = f'block{block}'
        self.computation_time = current_time - self.start_time_compute
        if key in self.output[epoch]['proc']:
            self.output[epoch]['proc'][key].append(duration)
            self.output[epoch]['compute'][key].append(self.computation_time)
        else:
            self.output[epoch]['proc'] = [duration]
            self.output[epoch]['compute']=[self.computation_time]
        self.logger.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size} samples in {duration:.4f}s)")

    def compute_metrics_train(self, epoch, block):
        key = f"block{block}"
        total_compute_time = np.sum(self.output[epoch]['compute'][key][self.metric_start_step:self.metric_end_step+1])
        total_time = self.end_timestamp - self.start_timestamp - np.sum(self.output[epoch]['proc'][key][:self.metric_start_step]) - np.sum(self.output[epoch]['proc'][key][self.metric_end_step+1:])
        if (total_compute_time==0):
            au=0.0
        else:
            au = total_compute_time / total_time
        throughput = (len(self.output[epoch]['compute'][key]) - 2)/(total_time)*self.batch_size
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def compute_metrics_eval(self, epoch):
        key = 'eval'
        total_compute_time = np.sum(self.output[epoch]['compute'][key][self.metric_start_step_eval:self.metric_end_step_eval+1])
        if (total_compute_time==0):
            au=0.0
        else:
            total_time = self.end_timestamp - self.start_timestamp - np.sum(self.output[epoch]['proc'][key][:self.metric_start_step_eval]) - np.sum(self.output[epoch]['proc'][key][self.metric_end_step_eval+1:])
            au = total_compute_time / total_time
        throughput = len(self.output[epoch]['compute'][key])/(self.end_timestamp - self.start_timestamp)*self.batch_size_eval
        self.output[epoch]['au'][key] = au*100
        self.output[epoch]['throughput'][key] = throughput

    def eval_batch_loaded(self, epoch, step):
        duration = time() - self.start_time_loading
        self.output[epoch]['load']['eval'].append(duration)
        self.logger.info(f"{utcnow()} Rank {self.my_rank} step {step} loaded {self.batch_size_eval} samples in {duration:.4f} s")

    def eval_batch_processed(self, epoch, step):
        current_time = time()
        duration = current_time - self.start_time_loading 
        computation_time = current_time - self.start_time_compute
        self.output[epoch]['proc']['eval'].append(duration)
        self.output[epoch]['compute']['eval'].append(computation_time)
        self.logger.info(f"{utcnow()} Rank {self.my_rank} step {step} processed {self.batch_size_eval} samples in {duration:.4f} s")
    def finalize(self):
        self.summary['end'] = utcnow()
    def save_data(self):
        # Dump statistic counters to files for postprocessing
        # Overall stats
        with open(os.path.join(self.output_folder, f'{self.my_rank}_per_epoch_stats.json'), 'w') as outfile:
            json.dump(self.per_epoch_stats, outfile, indent=4)
            outfile.flush()
        if self.my_rank == 0:
            with open(os.path.join(self.output_folder, 'summary.json'), 'w') as outfile:
                json.dump(self.summary, outfile, indent=4)
        self.output['hostname'] = socket.gethostname()
        with open(os.path.join(self.output_folder, f'{self.my_rank}_output.json'), 'w') as outfile:
            json.dump(self.output, outfile, indent=4)
            outfile.flush()
        if self.my_rank == 0:
            self.logger.output(f"{utcnow()} outputs saved in RANKID_output.json")
