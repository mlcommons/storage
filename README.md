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
git clone -b v1.0 --recurse-submodules https://github.com/mlcommons/storage.git
cd storage
pip3 install -r dlio_benchmark/requirements.txt
```

The working directory structure is as follows

```
|---storage
       |---benchmark.py
       |---benchmark
           |---(folder contains benchmark src files)
       |---configs
           |---dlio
               |---checkpointing
                   |---(folder contains configs for all checkpoint workloads)
               |---training
                   |---(folder contains configs of all training workloads)
           |---vectordbbench
               |---(folder contains configs for all vectordb workloads)
```

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) code, a benchmark suite for emulating I/O patterns for deep learning workloads. [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) currently is listed as a submodule to this MLPerf Storage repo. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `storage-conf` folder. ```benchmark.sh``` is a wrapper script which launches [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) to perform the benchmark for MLPerf Storage workloads. 

## Operation
The benchmarks uses nested commands to select the workload category, workload, and workload parameters.

### Workload Categories
The first command is the workload category
 - training
 - checkpointing
 - vectordb

```bash
./benchmark.py -h

usage: benchmark.py [-h] {training,checkpointing,vectordb} ...

Script to launch the MLPerf Storage benchmark

positional arguments:
  {training,checkpointing,vectordb}
                        Sub-programs
    training            Training benchmark options
    checkpointing       Checkpointing benchmark options
    vectordb            VectorDB benchmark options

optional arguments:
  -h, --help            show this help message and exit
```

### Training Category
The training category supports 3 models (unet3d, resnet50, cosmoflow). The benchmark execution process requires these steps:
1. Datasize - Calculate required number of samples for a given client configuration
2. Datagen - Generate the required dataset
3. Run - Execute the benchmark
4. Reportgen - Process the result logs into readable files

```bash
./benchmark.py training --help
usage: benchmark.py training [-h] [--debug] [--allow-invalid-params] [--stream-log-level STREAM_LOG_LEVEL]
                             {datasize,datagen,run,configview,reportgen} ...

positional arguments:
  {datasize,datagen,run,configview,reportgen}
                        Sub-commands
    datasize            The datasize command calculates the number of samples needed for a given workload,
                        accelerator type, number of accelerators, and client host memory.
    datagen             The datagen command generates a dataset for a given workload and number of parallel
                        generation processes.
    run                 Run the benchmark with the specified parameters.
    configview          View the final config based on the specified options.
    reportgen           Generate a report from the benchmark results.

optional arguments:
  -h, --help            show this help message and exit
  --debug               Enable debug mode
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.
  --stream-log-level STREAM_LOG_LEVEL
```

Use ```./benchmark.py training {command} --help``` for the full list of parameters for each command.

#### Data Sizing and Generation

**Note**: Steps described in this section must be run only in one client host(launcher client).

The datasize command relies on the accelerator being emulated, the max number of accelerators to support, the system memory in the benchmark clients, and the number of benchmark clients.

The two rules that generally dictate the datasize are:
1. The datasize on disk must be 5x the cumulative system memory of the benchmark clients
2. The benchmark must run for 500 iterations of the given batch size for all GPUs

If the list of clients is passed in for this command the amount of memory is found programmatically. Otherwise, the user can provide the number of clients and the amount of memory per client for the calculation.

```bash
./benchmark.py training datasize --help
usage: benchmark.py training datasize [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                                      --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB
                                      [--exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}] [--mpi-bin {mpirun,mpiexec}]
                                      [--oversubscribe] [--allow-run-as-root] --max-accelerators MAX_ACCELERATORS
                                      --accelerator-type {h100,a100} --num-client-hosts NUM_CLIENT_HOSTS
                                      [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                                      [--results-dir RESULTS_DIR] [--data-dir DATA_DIR] [--debug]
                                      [--allow-invalid-params] [--stream-log-level STREAM_LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and
                        data rates for each supported accelerator.
  --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB, -cm CLIENT_HOST_MEMORY_IN_GB
                        Memory available in the client where the benchmark is run. The dataset needs to be 5x the
                        available memory for closed submissions.
  --exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}, -et {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}
                        Execution type for benchmark commands. Supported options: [<EXEC_TYPE.MPI: 'mpi'>,
                        <EXEC_TYPE.DOCKER: 'docker'>]
  --max-accelerators MAX_ACCELERATORS, -ma MAX_ACCELERATORS
                        Max number of simulated accelerators. In multi-host configurations the accelerators will be
                        initiated in a round-robin fashion to ensure equal distribution of simulated accelerator
                        processes
  --accelerator-type {h100,a100}, -g {h100,a100}
                        Accelerator to simulate for the benchmark. A specific accelerator defines the data access
                        sizes and rates for each supported workload
  --num-client-hosts NUM_CLIENT_HOSTS, -nc NUM_CLIENT_HOSTS
                        Number of participating client hosts. Simulated accelerators will be initiated on these
                        hosts in a round-robin fashion
  --ssh-username SSH_USERNAME, -u SSH_USERNAME
                        Username for SSH for system information collection
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file.
                        For a closed submission only a subset of params are supported. Multiple values allowed in
                        the form: --params key1=value1 key2=value2 key3=value3
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --debug               Enable debug mode
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.
  --stream-log-level STREAM_LOG_LEVEL

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root

```

Example:

To calculate minimum dataset size for a `unet3d` model running on 2 client machines with 128 GB each with overall 8 simulated a100 accelerators

```bash
/benchmark.py training datasize -m unet3d --lient-host-memory-in-gb 128 --max-accelerators 16 --um-client-hosts 2 --accelerator-type a100  --results-dir ~/mlps-results
```

2. Synthetic data is generated based on the workload requested by the user.

```bash
./benchmark.py training datagen --help
usage: benchmark.py training datagen [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                                     --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB
                                     [--exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}] [--mpi-bin {mpirun,mpiexec}]
                                     [--oversubscribe] [--allow-run-as-root] --num-processes NUM_PROCESSES
                                     [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                                     [--results-dir RESULTS_DIR] [--data-dir DATA_DIR] [--debug]
                                     [--allow-invalid-params] [--stream-log-level STREAM_LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and
                        data rates for each supported accelerator.
  --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB, -cm CLIENT_HOST_MEMORY_IN_GB
                        Memory available in the client where the benchmark is run. The dataset needs to be 5x the
                        available memory for closed submissions.
  --exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}, -et {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}
                        Execution type for benchmark commands. Supported options: [<EXEC_TYPE.MPI: 'mpi'>,
                        <EXEC_TYPE.DOCKER: 'docker'>]
  --num-processes NUM_PROCESSES, -np NUM_PROCESSES
                        Number of parallel processes to use for dataset generation. Processes will be initiated in a
                        round-robin fashion across the configured client hosts
  --ssh-username SSH_USERNAME, -u SSH_USERNAME
                        Username for SSH for system information collection
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file.
                        For a closed submission only a subset of params are supported. Multiple values allowed in
                        the form: --params key1=value1 key2=value2 key3=value3
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --debug               Enable debug mode
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.
  --stream-log-level STREAM_LOG_LEVEL

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root
```

Example:

For generating training data of 56,000 files for `unet3d` workload into `unet3d_data` directory using 8 parallel jobs distributed on 2 nodes.

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --model unet3d --num-processes 8 --data-dir /mnt/unet3d_data --param dataset.num_files_train=56000
```

#### Running a Training Benchmark

```bash
./benchmark.py training run --help
usage: benchmark.py training run [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                                 --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB
                                 [--exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}] [--mpi-bin {mpirun,mpiexec}]
                                 [--oversubscribe] [--allow-run-as-root] --num-accelerators NUM_ACCELERATORS
                                 --accelerator-type {h100,a100} --num-client-hosts NUM_CLIENT_HOSTS
                                 [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                                 [--results-dir RESULTS_DIR] [--data-dir DATA_DIR] [--debug]
                                 [--allow-invalid-params] [--stream-log-level STREAM_LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and
                        data rates for each supported accelerator.
  --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB, -cm CLIENT_HOST_MEMORY_IN_GB
                        Memory available in the client where the benchmark is run. The dataset needs to be 5x the
                        available memory for closed submissions.
  --exec-type {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}, -et {EXEC_TYPE.MPI,EXEC_TYPE.DOCKER}
                        Execution type for benchmark commands. Supported options: [<EXEC_TYPE.MPI: 'mpi'>,
                        <EXEC_TYPE.DOCKER: 'docker'>]
  --num-accelerators NUM_ACCELERATORS, -na NUM_ACCELERATORS
                        Number of simulated accelerators. In multi-host configurations the accelerators will be
                        initiated in a round-robin fashion to ensure equal distribution of simulated accelerator
                        processes
  --accelerator-type {h100,a100}, -g {h100,a100}
                        Accelerator to simulate for the benchmark. A specific accelerator defines the data access
                        sizes and rates for each supported workload
  --num-client-hosts NUM_CLIENT_HOSTS, -nc NUM_CLIENT_HOSTS
                        Number of participating client hosts. Simulated accelerators will be initiated on these
                        hosts in a round-robin fashion
  --ssh-username SSH_USERNAME, -u SSH_USERNAME
                        Username for SSH for system information collection
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file.
                        For a closed submission only a subset of params are supported. Multiple values allowed in
                        the form: --params key1=value1 key2=value2 key3=value3
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --debug               Enable debug mode
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.
  --stream-log-level STREAM_LOG_LEVEL

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root

```

Example:

For running benchmark on `unet3d` workload with data located in `unet3d_data` directory using 2 h100 accelerators spread across 2 client hosts(with IPs 10.117.61.121,10.117.61.165) and results on `unet3d_results` directory, 

```bash
# TODO: Insert command to run unet3d workload 
```

4. Benchmark submission report is generated by aggregating the individual run results.

```bash
# TODO: Update
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

## Training Models
Currently, the storage benchmark suite supports benchmarking of 3 deep learning workloads
- Image segmentation using U-Net3D model 
- Image classification using Resnet-50 model
- Cosmology parameter prediction using CosmoFlow model

### U-Net3D

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload unet3d --accelerator-type h100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload unet3d --accelerator-type h100 --num-accelerators 2 --results-dir unet3d_h100 --param dataset.num_files_train=1200 --param dataset.data_folder=unet3d_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir unet3d_h100
```

### ResNet-50

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload resnet50 --accelerator-type h100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload resnet50 --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=resnet50_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload resnet50 --accelerator-type h100 --num-accelerators 2 --results-dir resnet50_h100 --param dataset.num_files_train=1200 --param dataset.data_folder=resnet50_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir resnet50_h100
```

### CosmoFlow

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
./benchmark.sh datasize --workload cosmoflow --accelerator-type h100 --num-accelerators 8 --num-client-hosts 2 --client-host-memory-in-gb 128
```

Generate data for the benchmark run

```bash
./benchmark.sh datagen --hosts 10.117.61.121,10.117.61.165 --workload cosmoflow --accelerator-type h100 --num-parallel 8 --param dataset.num_files_train=1200 --param dataset.data_folder=cosmoflow_data
```
  
Run the benchmark.

```bash
./benchmark.sh run --hosts 10.117.61.121,10.117.61.165 --workload cosmoflow --accelerator-type h100 --num-accelerators 2 --results-dir cosmoflow_h100 --param dataset.num_files_train=1200 --param dataset.data_folder=cosmoflow_data
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
./benchmark.sh reportgen --results-dir cosmoflow_h100
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
| reader.computation_threads    | Number of threads to preprocess the data(for TensorFlow)      |1|
| reader.prefetch_size    | Number of batches to prefetch      |2|
| reader.transfer_size       | Number of bytes in the read buffer(only for Tensorflow)  		        | |
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


## Submission Rules

MLPerf™ Storage Benchmark submission rules are described in this [doc](https://github.com/mlcommons/storage/blob/main/Submission_guidelines.md). If you have questions, please contact [Storage WG chairs](https://mlcommons.org/en/groups/research-storage/).
