# MLPerf Storage Benchmark Suite
MLPerf® Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

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
- 
## Overview
For an overview of how this benchmark suite is used by submitters to compare the performance of storage systems supporting an AI cluster, see the MLPerf® Storage Benchmark submission rules here: [doc](https://github.com/mlcommons/storage/blob/main/Submission_guidelines.md). 

## Prerequisite

The installation and the configuration steps described in this README are validated against clients running Ubuntu 24.04 server with python 3.12.3. The benchmark script has to be run only in one participating client host(any) which internally calls `mpirun` to launch the distributed workloads across multiple client hosts. The launcher client host also participates in the distributed training process.

Following prerequisites must be satisfied

1. Pick one host to act as the launcher client host. Passwordless ssh must be setup from the launcher client host to all other participating client hosts.  `ssh-copy-id` is a useful tool.
2. The code and data location(discussed in further sections) must be exactly same in every client host including the launcher host. This is because, the same benchmark command is automatically triggered in every participating client host during the distributed training process.

## Installation 
**The following installation steps must be run on every client host that will participate in running the benchmarks.**

### Dependencies
DLIO requires MPI package. 
For eg: when running on Ubuntu 24.04, install openmpi tools and libraries. 

```bash
sudo apt install python3-pip python3-venv libopenmpi-dev openmpi-common
```

Create a virtual environment for package installations and activate it.

```bash
python3 -m venv ~/.venvs/myenv
source ~/.venvs/myenv/bin/activate
```

### Pip
Please ensure you have the latest version of pip installed. This will fix the following error where the package is built as "UNKNOWN". Upgrade pip like so:

```bash
python3 -m pip install --upgrade pip
```


Clone the latest release from [MLCommons Storage](https://github.com/mlcommons/storage) repository and install Python dependencies.

```bash
git clone -b v2.0-rc0 https://github.com/mlcommons/storage.git
cd storage
pip3 install -e .
```

The working directory structure is as follows

```
|---storage
       |---mlpstorage
           |---(folder contains benchmark src files)
       |---configs
           |---dlio
               |---workload
                   |---(folder contains configs for all checkpoint and training workloads)
           |---vectordbbench (These configurations are PREVIEW only and not available for submission)
               |---(folder contains configs for all vectordb workloads)
```

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) code, a benchmark suite for emulating I/O patterns for deep learning workloads. [dlio_benchmark](https://github.com/argonne-lcf/dlio_benchmark) is listed as a prerequisite to a specific git branch. A future release will update the installer to pull DLIO from PyPi. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `configs` folder. 

## Operation
The benchmarks uses nested commands to select the workload category, workload, and workload parameters.

### Workload Categories
The first argument is the workload category
 - training
 - checkpointing
 - vectordb (PREVIEW)

```bash
[root@localhost ]#  mlpstorage -h
usage: mlpstorage [-h] [--version] {training,checkpointing,vectordb,reports,history} ...

Script to launch the MLPerf Storage benchmark

positional arguments:
  {training,checkpointing,vectordb,reports,history}
    training            Training benchmark options
    checkpointing       Checkpointing benchmark options
    vectordb            VectorDB benchmark options
    reports             Generate a report from benchmark results
    history             Display benchmark history

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### Training Category
The training category supports 3 models (unet3d, resnet50, cosmoflow). The benchmark execution process requires these steps:
1. Datasize - Calculate required number of samples for a given client configuration
2. Datagen - Generate the required dataset
3. Run - Execute the benchmark

```bash
[root@localhost ]# mlpstorage training --help
usage: mlpstorage training [-h] [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug] [--verbose]
                           [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params] [--what-if]
                           {datasize,datagen,run,configview} ...

Run the MLPerf Storage training benchmark

positional arguments:
  {datasize,datagen,run,configview}
    datasize            The datasize command calculates the number of samples needed for a given workload, accelerator
                        type, number of accelerators, and client host memory.
    datagen             The datagen command generates a dataset for a given workload and number of parallel generation
                        processes.
    run                 Run the benchmark with the specified parameters.
    configview          View the final config based on the specified options.

optional arguments:
  -h, --help            show this help message and exit

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.
```

Use ```mlpstorage training {command} --help``` for the full list of parameters for each command.

#### Data Sizing and Generation

**Note**: Steps described in this section must be run only in one client host(launcher client).

The datasize command relies on the accelerator being emulated, the max number of accelerators to support, the system memory in the benchmark clients, and the number of benchmark clients.

The two rules that generally dictate the datasize are:
1. The datasize on disk must be 5x the cumulative system memory of the benchmark clients
2. The benchmark must run for 500 iterations of the given batch size for all GPUs

If the list of clients is passed in for this command the amount of memory is found programmatically. Otherwise, the user can provide the number of clients and the amount of memory per client for the calculation.

```bash
[root@localhost ]# mlpstorage training datasize --help
usage: mlpstorage training datasize [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                                    --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB [--exec-type {mpi,docker}]
                                    [--mpi-bin {mpirun,mpiexec}] [--oversubscribe] [--allow-run-as-root]
                                    --max-accelerators MAX_ACCELERATORS --accelerator-type {h100,a100}
                                    --num-client-hosts NUM_CLIENT_HOSTS [--data-dir DATA_DIR]
                                    [--params PARAMS [PARAMS ...]]
                                    [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug]
                                    [--verbose] [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params]
                                    [--what-if]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and data
                        rates for each supported accelerator.
  --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB, -cm CLIENT_HOST_MEMORY_IN_GB
                        Memory available in the client where the benchmark is run. The dataset needs to be 5x the
                        available memory for closed submissions.
  --exec-type {mpi,docker}, -et {mpi,docker}
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
                        Number of participating client hosts. Simulated accelerators will be initiated on these hosts
                        in a round-robin fashion
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file. For
                        a closed submission only a subset of params are supported. Multiple values allowed in the
                        form: --params key1=value1 key2=value2 key3=value3
  --dlio-bin-path DLIO_BIN_PATH, -dp DLIO_BIN_PATH
                        Path to DLIO binary. Default is the same as mlpstorage binary path

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.
```

Example:

To calculate minimum dataset size for a `unet3d` model running on 2 client machines with 128 GB each with overall 8 simulated a100 accelerators

```bash
mlpstorage training datasize -m unet3d --client-host-memory-in-gb 128 --max-accelerators 16 --num-client-hosts 2 --accelerator-type a100  --results-dir ~/mlps-results
```

2. Synthetic data is generated based on the workload requested by the user.

```bash
[root@localhost ]# mlpstorage training datagen --help
usage: mlpstorage training datagen [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                                   [--exec-type {mpi,docker}] [--mpi-bin {mpirun,mpiexec}] [--oversubscribe]
                                   [--allow-run-as-root] --num-processes NUM_PROCESSES [--data-dir DATA_DIR]
                                   [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                                   [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug]
                                   [--verbose] [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params]
                                   [--what-if]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and data
                        rates for each supported accelerator.
  --exec-type {mpi,docker}, -et {mpi,docker}
                        Execution type for benchmark commands. Supported options: [<EXEC_TYPE.MPI: 'mpi'>,
                        <EXEC_TYPE.DOCKER: 'docker'>]
  --num-processes NUM_PROCESSES, -np NUM_PROCESSES
                        Number of parallel processes to use for dataset generation. Processes will be initiated in a
                        round-robin fashion across the configured client hosts
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file. For
                        a closed submission only a subset of params are supported. Multiple values allowed in the
                        form: --params key1=value1 key2=value2 key3=value3
  --dlio-bin-path DLIO_BIN_PATH, -dp DLIO_BIN_PATH
                        Path to DLIO binary. Default is the same as mlpstorage binary path

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.
```

Example:

For generating training data of 56,000 files for `unet3d` workload into `unet3d_data` directory using 8 parallel jobs distributed on 2 nodes.

```bash
mlpstorage training datagen --hosts 10.117.61.121,10.117.61.165 --model unet3d --num-processes 8 --data-dir /mnt/unet3d_data --param dataset.num_files_train=56000
```

#### Running a Training Benchmark

```bash
[root@localhost ]# mlpstorage training run --help
usage: mlpstorage training run [-h] [--hosts HOSTS [HOSTS ...]] --model {cosmoflow,resnet50,unet3d}
                               --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB [--exec-type {mpi,docker}]
                               [--mpi-bin {mpirun,mpiexec}] [--oversubscribe] [--allow-run-as-root] --num-accelerators
                               NUM_ACCELERATORS --accelerator-type {h100,a100} --num-client-hosts NUM_CLIENT_HOSTS
                               [--data-dir DATA_DIR] [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                               [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug] [--verbose]
                               [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params] [--what-if]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {cosmoflow,resnet50,unet3d}, -m {cosmoflow,resnet50,unet3d}
                        Model to emulate. A specific model defines the sample size, sample container format, and data
                        rates for each supported accelerator.
  --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB, -cm CLIENT_HOST_MEMORY_IN_GB
                        Memory available in the client where the benchmark is run. The dataset needs to be 5x the
                        available memory for closed submissions.
  --exec-type {mpi,docker}, -et {mpi,docker}
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
                        Number of participating client hosts. Simulated accelerators will be initiated on these hosts
                        in a round-robin fashion
  --data-dir DATA_DIR, -dd DATA_DIR
                        Filesystem location for data
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Additional parameters to be passed to the benchmark. These will override the config file. For
                        a closed submission only a subset of params are supported. Multiple values allowed in the
                        form: --params key1=value1 key2=value2 key3=value3
  --dlio-bin-path DLIO_BIN_PATH, -dp DLIO_BIN_PATH
                        Path to DLIO binary. Default is the same as mlpstorage binary path

MPI:
  --mpi-bin {mpirun,mpiexec}
                        Execution type for MPI commands. Supported options: ['mpirun', 'mpiexec']
  --oversubscribe
  --allow-run-as-root

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.

```

Example:

For running benchmark on `unet3d` workload with data located in `unet3d_data` directory using 2 h100 accelerators spread across 2 client hosts(with IPs 10.117.61.121,10.117.61.165) and results on `unet3d_results` directory, 

```bash
mlpstorage training run --hosts 10.117.61.121,10.117.61.165 --num-client-hosts 2 --client-host-memory-in-gb 64 --num-accelerators 2 --accelerator-type h100 --model unet3d  --data-dir unet3d_data --results-dir unet3d_results    --param dataset.num_files_train=400 
```

4. Benchmark submission report is generated by aggregating the individual run results. The reporting command provides the associated functions to generate a report for a given results directory

```bash
# TODO: Update
[root@localhost]# mlpstorage reports --help
usage: mlpstorage reports [-h] [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug] [--verbose]
                          [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params] [--what-if]
                          {reportgen} ...

positional arguments:
  {reportgen}           Sub-commands
    reportgen           Generate a report from the benchmark results.

optional arguments:
  -h, --help            show this help message and exit

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.
```

To generate the benchmark report,

```bash
[root@localhost]# mlpstorage reports reportgen --help
usage: mlpstorage reports reportgen [-h] [--output-dir OUTPUT_DIR] [--results-dir RESULTS_DIR] [--loops LOOPS]
                                    [--open | --closed] [--debug] [--verbose] [--stream-log-level STREAM_LOG_LEVEL]
                                    [--allow-invalid-params] [--what-if]

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory where the benchmark report will be saved.

Standard Arguments:
  --results-dir RESULTS_DIR, -rd RESULTS_DIR
                        Directory where the benchmark results will be saved.
  --loops LOOPS         Number of times to run the benchmark
  --open                Run as an open submission
  --closed              Run as a closed submission

Output Control:
  --debug               Enable debug mode
  --verbose             Enable verbose mode
  --stream-log-level STREAM_LOG_LEVEL
  --allow-invalid-params, -aip
                        Do not fail on invalid parameters.

View Only:
  --what-if             View the configuration that would execute and the associated command.
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
mlpstorage training datasize --model unet3d --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 4 --accelerator-type h100
```

Generate data for the benchmark run based on the minimum files

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model unet3d --data-dir unet3d_data --results-dir unet3d_results  --param dataset.num_files_train=42000
```
  
Run the benchmark.

```bash
mlpstorage training run --hosts 127.0.0.1 --num-client-hosts 1 --client-host-memory-in-gb 64 --num-accelerators 4 --accelerator-type h100 --model unet3d  --data-dir unet3d_data --results-dir unet3d_results --param dataset.num_files_train=42000 --checkpoint-folder unet3d_checkpoints
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir unet3d_results
```

### ResNet-50

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
 mlpstorage training datasize --model resnet50 --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 16 --accelerator-type h100
```

Generate data for the benchmark run

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model resnet50 --data-dir resnet50_data --results-dir resnet50_results  --param dataset.num_files_train=2557
```
  
Run the benchmark.

```bash
mlpstorage training run --hosts 127.0.0.1 --num-client-hosts 1  --client-host-memory-in-gb 64  --num-accelerators 16 --accelerator-type h100  --model resnet50  --data-dir resnet50_data --results-dir resnet50_results --param dataset.num_files_train=2557
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir resnet50_results
```

### CosmoFlow

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
mlpstorage training datasize --model cosmoflow --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 16 --accelerator-type h100 
```

Generate data for the benchmark run

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model cosmoflow --data-dir cosmoflow_data --results-dir=cosmoflow_results  --param dataset.num_files_train=121477
```
  
Run the benchmark.

```bash
mlpstorage training run  --hosts 127.0.0.1 --num-client-hosts 1  --client-host-memory-in-gb 64 --num-accelerators 16  --accelerator-type h100  --model cosmoflow --data-dir cosmoflow_data --results-dir cosmoflow_results --param dataset.num_files_train=121477 
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir cosmoflow_results
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
| reader.odirect                  | Whether to use direct I/O for reader (currectly applicable to UNet3D)   | False | 
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
