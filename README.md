# MLPerf™ Storage Benchmark Suite
MLPerf Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Configuration](#configuration)
- [Workloads](#workloads)
	- [U-Net3D](#u-net3d)
   	- [ResNet-50](#resnet-50)
   	- [CosmoFlow](#cosmoflow)
- [Parameters](#parameters)
	- [CLOSED](#closed)
	- [OPEN](#open)
- [Submission Rules](#submission-rules)
## Overview

This section describes how to use the MLPerf™ Storage Benchmark to measure the performance of a storage system supporting a compute cluster running AI/ML training tasks.
 
This benchmark attempts to balance two goals:
1.	Comparability between benchmark submissions to enable decision making by the AI/ML Community.
2.	Flexibility to enable experimentation and to show off unique storage system features that will benefit the AI/ML Community.
 
To that end we have defined two classes of submissions: CLOSED and OPEN.
 
CLOSED represents a level playing field where all<sup>*</sup> results are comparable across submissions.  CLOSED explicitly forfeits flexibility in order to enable easy comparability. 

<sup>*</sup> The benchmark supports both PyTorch and TensorFlow data formats, however these formats substantially different loads to the storage system such that cross-format comparisons are not appropriate, even with CLOSED submissions.  Therefore only comparisons of storage systems using the same data format are valid (e.g., two CLOSED PyTorch runs or two CLOSED TensorFlow runs.  As new data formats like PyTorch and TensorFlow are added to the benchmark that categorization will grow.
 
OPEN allows more flexibility to tune and change both the benchmark and the storage system configuration to show off new approaches or new features that will benefit the AI/ML Community.  OPEN explicitly forfeits comparability to allow showcasing innovation.

**Benchmark output metric**

For each workload, the benchmark output metric is samples per second, subject to a minimum *accelerator utilization* (```AU```), where higher is better. To pass a benchmark run, ```AU``` should be 90% or higher. ```AU``` is computed as follows. The total ideal compute time is derived from the batch size, total dataset size, number of simulated accelerators, and sleep time: ```total_compute_time = (records/file * total_files)/simulated_accelerators/batch_size * sleep_time```. Then ```AU``` is computed as follows: 

```
AU (percentage) = (total_compute_time/total_benchmark_running_time) * 100
```

Note that the sleep time has been determined by running the workloads including the compute step on real hardware and is dependent on the accelerator type. In this preview package we include sleep times for NVIDIA V100 GPUs, as measured in an NVIDIA DGX-1 system.

In addition to ```AU```, submissions are expected to report details such as the number of MPI processes run on the DLIO host, as well as the amount of main memory on the DLIO host.

**Future work**

In a future version of the benchmark, the MLPerf Storage WG plans to add support for the “data preparation” phase of AI/ML workload as we believe that is a significant load on a storage system and is not well represented by existing AI/ML benchmarks, but the current version only requires a static copy of the dataset exist on the storage system before the start of the run.
 
In a future version of the benchmark, the MLPerf Storage WG plans to add support for benchmarking a storage system while running more than one MLPerf Storage benchmark at the same time (ie: more than one Training job type, such as 3DUnet and Recommender at the same time), but the current version requires that a submission only include one such job type per submission.

In a future version of the benchmark, we aim to include sleep times for different accelerator types, including different types of GPUs and other ASICS.

## Prerequisite

The installation and the configuration steps described in this README are validated against clients running Ubuntu 22.04 server with python 3.10.12. The benchmark script has to be run only in one participating client host(any) which internally calls `mpirun` to launch the distributed training across multiple client hosts. The launcher client host also participates in the distributed training process.

Following prerequisites must be satisfied

1. Pick one host to act as the launcher client host. Passwordless ssh must be setup from the launcher client host to all other participating client hosts.  `ssh-copy-id` is a useful tool.
2. The code and data location(discussed in further sections) must be exactly same in every client host including the launcher host. This is because, the same benchmark command is automatically triggered in every participating client host during the distributed training process.

## Installation 

**Note**: Steps described in this sections must be run in every client host.

Install dependencies using your system package manager.
- `mpich` for MPI package

For eg: when running on Ubuntu OS,

```
sudo apt-get install mpich
```

Clone the latest release from [MLCommons Storage](https://github.com/mlcommons/storage) repository and install Python dependencies.

```bash
git clone -b v1.0-rc1 --recurse-submodules https://github.com/mlcommons/storage.git
cd storage
pip3 install -r dlio_benchmark/requirements.txt
```

The working directory structure is as follows

```
|---storage
       |---benchmark.sh
       |---dlio_benchmark
       |---storage-conf
           |---workload(folder contains configs of all workloads)

```

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) code, a benchmark suite for emulating I/O patterns for deep learning workloads. [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) currently is listed as a submodule to this MLPerf Storage repo. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `storage-conf` folder. ```benchmark.sh``` is a wrapper script which launches [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) to perform the benchmark for MLPerf Storage workloads. 

```bash
./benchmark.sh -h

Usage: ./benchmark.sh [datasize/datagen/run/configview/reportgen] [options]
Script to launch the MLPerf Storage benchmark.
```

## Configuration

**Note**: Steps described in this section must be run only in one client host(launcher client).

The benchmark suite consists of 4 distinct phases

1. Calculate the minimum dataset size required for the benchmark run

```bash
./benchmark.sh datasize -h
Usage: ./benchmark.sh datasize [options]
Get minimum dataset size required for the benchmark run.


Options:
  -h, --help			        Print this message
  -w, --workload		        Workload dataset to be generated. Possible options are 'unet3d', 'cosmoflow' 'resnet50'
  -g, --accelerator-type	        Simulated accelerator type used for the benchmark. Possible options are 'a100' 'h100'
  -n, --num-accelerators	        Simulated number of accelerators(of same accelerator type)
  -c, --num-client-hosts	        Number of participating client hosts
  -m, --client-host-memory-in-gb	Memory available in the client where benchmark is run
```

Example:

To calculate minimum dataset size for a `unet3d` workload running on 2 client machines with 128 GB each with overall 8 simulated a100 accelerators

```bash
./benchmark.sh datasize --workload unet3d --accelerator-type a100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

2. Synthetic data is generated based on the workload requested by the user.

```bash
./benchmark.sh datagen -h

Usage: ./benchmark.sh datagen [options]
Generate benchmark dataset based on the specified options.


Options:
  -h, --help			Print this message
  -s, --hosts			Comma separated IP addresses of the participating hosts(without space). eg: '192.168.1.1,192.168.2.2'
  -c, --category		Benchmark category to be submitted. Possible options are 'closed'(default)
  -w, --workload		Workload dataset to be generated. Possible options are 'unet3d', 'cosmoflow' 'resnet50'
  -g, --accelerator-type	Simulated accelerator type used for the benchmark. Possible options are 'a100' 'h100'
  -n, --num-parallel		Number of parallel jobs used to generate the dataset
  -r, --results-dir		Location to the results directory. Default is ./results/workload.model/DATE-TIME
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For generating training data for `unet3d` workload into `unet3d_data` directory using 8 parallel jobs distributed on 2 nodes for h100 simulated accelerator,

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```

3. Benchmark is run on the generated data.

```bash
./benchmark.sh run -h

Usage: ./benchmark.sh run [options]
Run benchmark on the generated dataset based on the specified options.


Options:
  -h, --help			Print this message
  -s, --hosts			Comma separated IP addresses of the participating hosts(without space). eg: '192.168.1.1,192.168.2.2'
  -c, --category		Benchmark category to be submitted. Possible options are 'closed'(default)
  -w, --workload		Workload to be run. Possible options are 'unet3d', 'cosmoflow' 'resnet50'
  -g, --accelerator-type	Simulated accelerator type used for the benchmark. Possible options are 'a100' 'h100'
  -n, --num-accelerators	Simulated number of accelerators(of same accelerator type)
  -r, --results-dir		Location to the results directory.
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For running benchmark on `unet3d` workload with data located in `unet3d_data` directory using 2 h100 accelerators spread across 2 client hosts(with IPs 10.117.61.121,10.117.61.165) and results on `unet3d_results` directory, 

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-accelerators 2 --results-dir resultsdir --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```

4. Benchmark submission report is generated by aggregating the individual run results.

```bash
./benchmark.sh reportgen -h

Usage: ./benchmark.sh reportgen [options]
Generate a report from the benchmark results.


Options:
  -h, --help			Print this message
  -r, --results-dir		Location to the results directory
```
To generate the benchmark report,

```bash
./benchmark.sh reportgen --results-dir  resultsdir
```

Note: The `reportgen` script must be run in the launcher client host. 

## Workloads
Currently, the storage benchmark suite supports benchmarking of 3 deep learning workloads
- Image segmentation using U-Net3D model 
- Image classification using Resnet-50 model
- Cosmology parameter prediction using CosmoFlow model

### U-Net3D

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload unet3d --accelerator-type a100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-accelerators 2 --results-dir resultsdir --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir resultsdir
```

### ResNet-50

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload resnet50 --accelerator-type a100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload resnet50 --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=resnet50_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload resnet50 --accelerator-type h100 --num-accelerators 2 --results-dir resultsdir --param dataset.num_files_train=1200 --param dataset.data_folder=resnet50_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir resultsdir
```

### CosmoFlow

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload cosmoflow --accelerator-type a100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload cosmoflow --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=cosmoflow_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload cosmoflow --accelerator-type h100 --num-accelerators 2 --results-dir resultsdir --param dataset.num_files_train=1200 --param dataset.data_folder=cosmoflow_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir resultsdir
```

## Parameters 

### CLOSED
Below table displays the list of configurable parameters for the benchmark in the closed category.

| Parameter                      | Description                                                 |Default|
| ------------------------------ | ------------------------------------------------------------ |-------|
| **Dataset params**		|								|   |
| dataset.num_files_train       | Number of files for the training set  		        | --|
| dataset.num_subfolders_train  | Number of subfolders that the training set is stored	        |0|
| dataset.data_folder           | The path where dataset is stored				| --|
| **Reader params**				|						|   |
| reader.read_threads		| Number of threads to load the data                            | --|
| reader.computation_threads    | Number of threads to preprocess the data(for TensorFlow)      | --|
| **Checkpoint params**		|								|   |
| checkpoint.checkpoint_folder	| The folder to save the checkpoints  				| --|
| **Storage params**		|								|   |
| storage.storage_root		| The storage root directory  					| ./|
| storage.storage_type		| The storage type  						|local_fs|


### OPEN
In addition to what can be changed in the CLOSED category, the following parameters can be changed in the OPEN category.

| Parameter                      | Description                                                 |Default|
| ------------------------------ | ------------------------------------------------------------ |-------|
| framework		| The machine learning framework		|Pytorch for 3D U-Net |
| **Dataset params**		|								|   |
| dataset.format       | Format of the dataset  		        | .npz for 3D U-Net |
| dataset.num_samples_per_file       | Number of samples per file(only for Tensorflow using tfrecord datasets)  		        | 1 for 3D U-Net |
| **Reader params**		|
| reader.data_loader       | Data loader type(Tensorflow or PyTorch or custom) 		        | PyTorch for 3D U-Net |
| reader.transfer_size       | Number of bytes in the read buffer(only for Tensorflow)  		        | |

## Submission Rules

MLPerf™ Storage Benchmark submission rules are described in this [doc](https://github.com/mlcommons/storage/blob/main/Submission_guidelines.md). If you have questions, please contact [Storage WG chairs](https://mlcommons.org/en/groups/research-storage/).
