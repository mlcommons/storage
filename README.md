# MLPerf™ Storage Benchmark Suite
MLPerf Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Workloads](#workloads)
	- [U-Net3D](#u-net3d)
	- [BERT](#bert)
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


## Installation 

Install dependencies using your system package manager.
- `mpich` for MPI package

For eg: when running on Ubuntu OS,

```
sudo apt-get install mpich
```

Clone the latest release from [MLCommons Storage](https://github.com/mlcommons/storage) repository and install Python dependencies.

```bash
git clone -b v0.5-rc2 --recurse-submodules https://github.com/mlcommons/storage.git
cd storage
pip3 install -r dlio_benchmark/requirements.txt
```

The working directory structure is as follows

```
|---storage
       |---benchmark.sh
	   |---report.py
       |---dlio_benchmark
       |---storage-conf
           |---workload(folder contains configs of all workloads)
               |---unet3d.yaml
               |---bert.yaml
```

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) code, a benchmark suite for emulating I/O patterns for deep learning workloads. [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) currently is listed as a submodule to this MLPerf Storage repo. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `storage-conf` folder. ```benchmark.sh``` is a wrapper script which launches [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) to perform the benchmark for MLPerf Storage workloads. 

```bash
./benchmark.sh -h

Usage: ./benchmark.sh [datasize/datagen/run/configview/reportgen] [options]
Script to launch the MLPerf Storage benchmark.
```

## Configuration

The benchmark suite consists of 4 distinct phases

1. Calculate the minimum dataset size required for the benchmark run

```bash
./benchmark.sh datasize -h
Usage: ./benchmark.sh datasize [options]
Get minimum dataset size required for the benchmark run.


Options:
  -h, --help			Print this message
  -w, --workload		Workload dataset to be generated. Possible options are 'unet3d', 'bert'
  -n, --num-accelerators	Simulated number of accelerators per node of same accelerator type
  -m, --host-memory-in-gb	Memory available in the client where benchmark is run
```

Example:

To calculate minimum dataset size for a `unet3d` workload on a client machine with 128 GB with 8 simulated accelerators,

```bash
./benchmark.sh datasize --workload unet3d --num-accelerators 8 --host-memory-in-gb 128
```

2. Synthetic data is generated based on the workload requested by the user.

```bash
./benchmark.sh datagen -h

Usage: ./benchmark.sh datagen [options]
Generate benchmark dataset based on the specified options.


Options:
  -h, --help			Print this message
  -c, --category		Benchmark category to be submitted. Possible options are 'closed'(default)
  -w, --workload		Workload dataset to be generated. Possible options are 'unet3d', 'bert'
  -n, --num-parallel		Number of parallel jobs used to generate the dataset
  -r, --results-dir		Location to the results directory. Default is ./results/workload.model/DATE-TIME
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For generating training data for `unet3d` workload into `unet3d_data` directory with 10 subfolders using 8 parallel jobs, 

```bash
./benchmark.sh datagen --workload unet3d --num-parallel 8 --param dataset.num_subfolders_train=10 --param dataset.data_folder=unet3d_data
```

3. Benchmark is run on the generated data.

```bash
./benchmark.sh run -h

Usage: ./benchmark.sh run [options]
Run benchmark on the generated dataset based on the specified options.


Options:
  -h, --help			Print this message
  -c, --category		Benchmark category to be submitted. Possible options are 'closed'(default)
  -w, --workload		Workload to be run. Possible options are 'unet3d', 'bert'
  -g, --accelerator-type	Simulated accelerator type used for the benchmark. Possible options are 'v100-32gb'(default)
  -n, --num-accelerators	Simulated number of accelerators of same accelerator type
  -r, --results-dir		Location to the results directory. Default is ./results/workload.model/DATE-TIME
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For running benchmark on `unet3d` workload with data located in `unet3d_data` directory using 4 accelerators and results on `unet3d_results` directory , 

```bash
./benchmark.sh run --workload unet3d --num-accelerators 4 --results-dir unet3d_results --param dataset.data_folder=unet3d_data
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

The result directory needs to be in the following structure which must include 5 runs.

```
sample-results
	|---run-1
	       |---host-1
	                |---summary.json
	       |---host-2
	                |---summary.json
	          ....
	       |---host-n
	                |---summary.json
	|---run-2
	       |---host-1
 	               |---summary.json
	       |---host-2
	                |---summary.json
	          ....
 	       |---host-n
 	               |---summary.json
	    .....
	|---run-5
	       |---host-1
	                |---summary.json
	       |---host-2
 	               |---summary.json
 	          ....
 	       |---host-n
 	               |---summary.json
```

To generate the benchmark report,

```bash
./benchmark.sh reportgen --results-dir  sample-results/
```

For reference, a sample result directory structure can be found [here](https://github.com/johnugeorge/mlperf-storage-sample-results). 

## Workloads
Currently, the storage benchmark suite supports benchmarking of 2 deep learning workloads
- Image segmentation using U-Net3D model ([unet3d](./storage-conf/workload/unet3d.yaml))
- Natural language processing using BERT model ([bert](./storage-conf/workload/bert.yaml))

### U-Net3D

Calculate minimum dataset size required for the benchmark run

```bash
./benchmark.sh datasize --workload unet3d --num-accelerators 8 --host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --workload unet3d --num-parallel 8 --param dataset.num_files_train=3200
```
  
Run the benchmark.

```bash
./benchmark.sh run --workload unet3d --num-accelerators 8 --param dataset.num_files_train=3200
```

All results will be stored in ```results/unet3d/$DATE-$TIME``` folder or in the directory when overridden using `--results-dir`(or `-r`) argument. To generate the final report, one can do

```bash 
./benchmark.sh reportgen --results-dir results/unet3d/$DATE-$TIME
```
This will generate ```mlperf_storage_report.json``` in the output folder.

### BERT

Calculate minimum dataset size required for the benchmark run

```bash
./benchmark.sh datasize --workload bert --num-accelerators 8 --host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --workload bert --num-parallel 8 --param dataset.num_files_train=350
```
  
Run the benchmark
```bash
./benchmark.sh run --workload bert --num-accelerators 8 --param dataset.num_files_train=350
```

All results will be stored in ```results/bert/$DATE-$TIME``` folder or in the directory when overridden using `--results-dir`(or `-r`) argument. To generate the final report, one can do

```bash 
./benchmark.sh reportgen -r results/bert/$DATE-$TIME
```
This will generate ```mlperf_storage_report.json``` in the output folder.

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
| reader.computation_threads    | Number of threads to preprocess the data(only for Bert)       | --|
| **Checkpoint params**		|								|   |
| checkpoint.checkpoint_folder	| The folder to save the checkpoints  				| --|
| **Storage params**		|								|   |
| storage.storage_root		| The storage root directory  					| ./|
| storage.storage_type		| The storage type  						|local_fs|


### OPEN
In addition to what can be changed in the CLOSED category, the following parameters can be changed in the OPEN category.

| Parameter                      | Description                                                 |Default|
| ------------------------------ | ------------------------------------------------------------ |-------|
| framework		| The machine learning framework		|Pytorch for 3D U-Net, Tensorflow for Bert |
| **Dataset params**		|								|   |
| dataset.format       | Format of the dataset  		        | .npz for 3D U-Net and tfrecord for Bert|
| dataset.num_samples_per_file       | Number of samples per file(only for Tensorflow using tfrecord datasets)  		        | For 3D U-Net: 1 and for Bert: 313532|
| **Reader params**		|
| reader.data_loader       | Data loader type(Tensorflow or PyTorch or custom) 		        | PyTorch for 3D U-Net, and Tensorflow for Bert|
| reader.transfer_size       | Number of bytes in the read buffer(only for Tensorflow)  		        | For BERT: 262144|

## Submission Rules

MLPerf™ Storage Benchmark submission rules are described in this [doc](https://docs.google.com/document/d/1QOaCLiWb82H9cwdVX5KyeDZWt0781y4SgMQPhoij-b4/edit). If you have questions, please contact [Storage WG chairs](https://mlcommons.org/en/groups/research-storage/).
