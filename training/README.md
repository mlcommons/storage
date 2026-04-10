# MLPerf Storage Benchmark Suite - Training Workloads
MLPerf® Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Usage](#Usage)
  - [Workloads](#workloads)
	  - [FLUX.1](#flux-1)
    - [RetinaNet](#retinanet)
    - [DLRMv2](#dlrmv2)
  - [Parameters](#parameters)
  	- [CLOSED](#closed)
  	- [OPEN](#open)
- [Theory of Operations](#theory-of-operations)
  - [Benchmark Overview](#benchmark-overview)
  - [Definitions](#definitions)
  - [Performance Metrics](#performance-metrics)
  - [Dataset Generation](#dataset-generation)
  - [Single-host Submissions](#single-host-submissions)
  - [Multi-host (Distributed) Training Submissions](#multi-host-distributed-training-submissions)
  - [CLOSED and OPEN Divisions](#closed-and-open-divisions)

---

# Usage
The training category supports 3 models (FLUX.1, RetinaNet, DLRMv2).
The benchmark execution process requires these steps:
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
usage: mlpstorage training datasize [-h] [--hosts HOSTS [HOSTS ...]] --model {FLUX.1,RetinaNet,DLRMv2}
                                    --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB [--exec-type {mpi,docker}]
                                    [--mpi-bin {mpirun,mpiexec}] [--oversubscribe] [--allow-run-as-root]
                                    --max-accelerators MAX_ACCELERATORS --accelerator-type {b200,mi355}
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
  --model {FLUX.1,RetinaNet,DLRMv2}, -m {FLUX.1,RetinaNet,DLRMv2}
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
  --accelerator-type {b200,mi355}, -g {b200,mi355}
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

To calculate minimum dataset size for a `retinanet` model running on 2 client machines with 128 GB each with overall 8 simulated b200 accelerators

```bash
mlpstorage training datasize -m retinanet --client-host-memory-in-gb 128 --max-accelerators 16 --num-client-hosts 2 --accelerator-type a100  --results-dir ~/mlps-results
```

2. Synthetic data is generated based on the workload requested by the user.

```bash
[root@localhost ]# mlpstorage training datagen --help
usage: mlpstorage training datagen [-h] [--hosts HOSTS [HOSTS ...]] --model {flux1,dlrmv2,retinanet}
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
  --model {flux1,dlrmv2,retinanet}, -m {flux1,dlrmv2,retinanet}
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

For generating training data of 56,000 files for `retinanet` workload into `unet3d_data` directory using 8 parallel jobs distributed on 2 nodes.

```bash
mlpstorage training datagen --hosts 10.117.61.121,10.117.61.165 --model retinanet --num-processes 8 --data-dir /mnt/unet3d_data --param dataset.num_files_train=56000
```

#### Running a Training Benchmark

```bash
[root@localhost ]# mlpstorage training run --help
usage: mlpstorage training run [-h] [--hosts HOSTS [HOSTS ...]] --model {flux1,dlrmv2,retinanet}
                               --client-host-memory-in-gb CLIENT_HOST_MEMORY_IN_GB [--exec-type {mpi,docker}]
                               [--mpi-bin {mpirun,mpiexec}] [--oversubscribe] [--allow-run-as-root] --num-accelerators
                               NUM_ACCELERATORS --accelerator-type {b200,mi355} --num-client-hosts NUM_CLIENT_HOSTS
                               [--data-dir DATA_DIR] [--ssh-username SSH_USERNAME] [--params PARAMS [PARAMS ...]]
                               [--results-dir RESULTS_DIR] [--loops LOOPS] [--open | --closed] [--debug] [--verbose]
                               [--stream-log-level STREAM_LOG_LEVEL] [--allow-invalid-params] [--what-if]

optional arguments:
  -h, --help            show this help message and exit
  --hosts HOSTS [HOSTS ...], -s HOSTS [HOSTS ...]
                        Space-separated list of IP addresses or hostnames of the participating hosts. Example: '--
                        hosts 192.168.1.1 192.168.1.2 192.168.1.3' or '--hosts host1 host2 host3'
  --model {flux1,dlrmv2,retinanet}, -m {flux1,dlrmv2,retinanet}
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
  --accelerator-type {b200,mi355}, -g {b200,mi355}
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

For running benchmark on `retinanet` workload with data located in `unet3d_data` directory using 2 b200 accelerators spread across 2 client hosts(with IPs 10.117.61.121,10.117.61.165) and results on `unet3d_results` directory, 

```bash
mlpstorage training run --hosts 10.117.61.121,10.117.61.165 --num-client-hosts 2 --client-host-memory-in-gb 64 --num-accelerators 2 --accelerator-type b200 --model retinanet  --data-dir unet3d_data --results-dir unet3d_results    --param dataset.num_files_train=400 
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
- Image generation using a FLUX.1 model 
- Image recognition using a RetinaNet model
- Recommendations using a DLRMv2 model

### FLUX.1

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
mlpstorage training datasize --model retinanet --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 4 --accelerator-type b200
```

Generate data for the benchmark run based on the minimum files

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model retinanet --data-dir retinanet_data --results-dir retinanet_results  --param dataset.num_files_train=42000
```
  
Run the benchmark.

```bash
mlpstorage training run --hosts 127.0.0.1 --num-client-hosts 1 --client-host-memory-in-gb 64 --num-accelerators 4 --accelerator-type b200 --model retinanet  --data-dir retinanet_data --results-dir retinanet_results --param dataset.num_files_train=42000
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir retinanet_results
```

### RetinaNet

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
 mlpstorage training datasize --model dlrmv2 --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 16 --accelerator-type b200
```

Generate data for the benchmark run

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model dlrmv2 --data-dir dlrmv2_data --results-dir dlrmv2_results  --param dataset.num_files_train=2557
```
  
Run the benchmark.

```bash
mlpstorage training run --hosts 127.0.0.1 --num-client-hosts 1  --client-host-memory-in-gb 64  --num-accelerators 16 --accelerator-type b200  --model dlrmv2  --data-dir dlrmv2_data --results-dir dlrmv2_results --param dataset.num_files_train=2557
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir dlrmv2_results
```

### DLRMv2

Calculate minimum dataset size required for the benchmark run based on your client configuration

```bash
mlpstorage training datasize --model flux1 --client-host-memory-in-gb 64 --num-client-hosts 1 --max-accelerators 16 --accelerator-type b200 
```

Generate data for the benchmark run

```bash
mlpstorage training datagen --hosts 127.0.0.1 --num-processes 8 --model flux1 --data-dir flux1_data --results-dir=flux1_results  --param dataset.num_files_train=121477
```
  
Run the benchmark.

```bash
mlpstorage training run  --hosts 127.0.0.1 --num-client-hosts 1  --client-host-memory-in-gb 64 --num-accelerators 16  --accelerator-type b200  --model flux1 --data-dir flux1_data --results-dir flux1_results --param dataset.num_files_train=121477 
```

All results will be stored in the directory configured using `--results-dir`(or `-r`) argument. To generate the final report, run the following in the launcher client host. 

```bash 
mlpstorage reports reportgen --results-dir flux1_results
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
| reader.odirect                  | Whether to use direct I/O for reader   | False | 
| **Checkpoint params**		|								|   |
| checkpoint.checkpoint_folder	| The folder to save the checkpoints  				| --|
| **Storage params**		|								|   |
| storage.storage_root		| The storage root directory  					| ./|
| storage.storage_type		| The storage type  						|local_fs|


### OPEN
In addition to what can be changed in the CLOSED category, the following parameters can be changed in the OPEN category.

| Parameter                      | Description                                                 |Default|
| ------------------------------ | ------------------------------------------------------------ |-------|
| framework		                   | The machine learning framework		|Pytorch for 3D U-Net |
| **Dataset params**	        	 |								|   |
| dataset.format                 | Format of the dataset  		      | .npz for 3D U-Net |
| dataset.num_samples_per_file   | Number of samples per file(only for Tensorflow using tfrecord datasets)  		        | 1 for 3D U-Net |
| **Reader params**		           |
| reader.data_loader             | Data loader type(Tensorflow or PyTorch or custom) 		        | PyTorch for 3D U-Net |



# Theory of Operations
MLPerf™ Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads. The suite consists of 4 workload categories:

1. Training
2. Checkpointing
3. Vector Database
4. KVCache

This benchmark attempts to balance two goals. First, we aim for **comparability** between benchmark submissions to enable decision making by the AI/ML Community. Second, we aim for **flexibility** to enable experimentation and to show off unique storage system features that will benefit the AI/ML Community. To that end we have defined two classes of submissions: CLOSED and OPEN. 

The MLPerf name and logo are trademarks of the MLCommons® Association ("MLCommons"). In order to refer to a result using the MLPerf name, the result must conform to the letter and spirit of the rules specified in this document. MLCommons reserves the right to solely determine if a use of its name or logos is acceptable.

## Benchmark Overview

This version of the benchmark does not include offline or online data pre-processing. We are aware that data pre-processing is an important part of the ML data pipeline and we will include it in a future version of the benchmark.

MLPerf Storage emulates (or "simulates", the terms are used interchangably in this document) accelerators for the training workloads with the tool DLIO developed by Argonne National Labs. DLIO uses the standard AI frameworks (PyTorch, Tensorflow, Numpy, etc) to load data from storage to memory at the same intensity as a given accelerator.

**This emulation means that submitters do not need to use hardware accelerators (e.g., GPUs, TPUs, and other ASICs) when running MLPerf Storage - Training.**

Instead, our benchmark tool replaces the training on the accelerator for a single batch of data with a ``sleep()`` call. The ``sleep()`` interval depends on the batch size and accelerator type and has been determined through measurement on a system running the real training workload. The rest of the data ingestion pipeline (data loading, caching, checkpointing) is unchanged and runs in the same way as when the actual training is performed.

There are two main advantages to accelerator emulation. First, MLPerf Storage allows testing different storage systems with different types of accelerators. To change the type of accelerator that the benchmark emulates (e.g., to switch to a system with NVIDIA B200 GPUs), it is enough to adjust the batch size and ``sleep()`` parameter. The second advantage is that MLPerf Storage can put a high load on the storage system simply by increasing the number of emulated accelerators. This allows for testing the behavior of the storage system in large-scale scenarios without purchasing/renting the AI compute infrastructure.

The benchmark suite provides workload [configurations](https://github.com/mlcommons/storage/tree/main/storage-conf/workload) that simulate the I/O patterns of selected workloads listed in Table 1. The I/O patterns for each MLPerf Storage benchmark correspond to the I/O patterns of the MLPerf Training and MLPerf HPC benchmarks (i.e., the I/O generated by our tool for 3D U-Net closely follows the I/O generated by actually running the 3D U-Net training workload). The benchmark suite can also generate synthetic datasets which show the same I/O load as the actual datasets listed in Table 1. 

| Area | Problem | Model | Data Loader | Dataset seed | Minimum AU% |
| ---- | ------- | ----- | ----------- | ------------ | ----------- |
| Vision | Image generation | FLUX.1 | PyTorch | ??? | 90% |
| Vision | Image Recognition | RetinaNet | PyTorch | ??? | 90% |
| Recommender | Recommender | ??? | PyTorch | ??? | 90% |

Table 1: Benchmark description

- Benchmark start point: The dataset is in **shared persistent storage**. 
- Benchmark end point: The measurement ends after a predetermined number of epochs. *Note: data transfers from storage in this test terminate with the data in host DRAM; transfering data into the accelerator memory is not included in this benchmark.*
- Configuration files for the workloads and dataset content can be found [here](https://github.com/mlcommons/storage/tree/main/storage-conf/workload).

## Definitions 
The following definitions are used throughout this document:

- A **sample** is the unit of data on which training is run, e.g., an image, or a sentence.
- A **step** is defined to be the first batch of data loaded into the (emulated) accelerator.
- **Accelerator Utilization (AU)** is defined as the percentage of time taken by the simulated accelerators, relative to the total benchmark running time. Higher is better.
- **Design power** is defined to be the minimum measurement of electrical power that must be capable of being supplied to a single or collection of power supply units (PSUs) in order to avoid violating regulatory and safety requirements. For individual PSUs, the design power equals the nameplate rated power. For groups of redundant PSUs, the design power is equal to the sum of the nameplate rated power of the minimum number of PSUs required to be simultaneously operational.
- A **division** is a set of rules for implementing benchmarks from a suite to produce a class of comparable results. MLPerf Storage allows CLOSED and OPEN divisions, detailed in Section 6.
- **DLIO ([code link](https://github.com/argonne-lcf/dlio_benchmark), [paper link](https://ieeexplore.ieee.org/document/9499416))** is a benchmarking tool for deep learning applications. DLIO is the core of the MLPerf Storage benchmark and with specified configurations will emulate the I/O pattern for the workloads listed in Table 1.  MLPerf Storage provides wrapper scripts to launch DLIO. There is no need to know the internals of DLIO to do a CLOSED submission, as the wrapper scripts provided by MLPerf Storage will suffice. However, for OPEN submissions changes to the DLIO code might be required (e.g., to add custom data loaders). 
- **Dataset content** refers to the data and the total capacity of the data, not the format of how the data is stored. Specific information on dataset content can be found [here](https://github.com/mlcommons/storage/tree/main/storage-conf/workload). 
- **Dataset format** refers to the format in which the training data is stored (e.g., npz, hdf5, csv, png, tfrecord, etc.), not the content or total capacity of the dataset.

  *NOTE: we plan to add support for Object storage in a future version of the benchmark, so OPEN submissions that include benchmark application changes and a description of how the original MLPerf Training benchmark dataset was mapped into Objects will be appreciated.*
- A **storage system** consists of a defined set of hardware and software resources that provide storage services to one or more ``host nodes``. Storage systems can be hardware based, software-defined, virtualized, hyperconverged, or cloud based, and must be capable of providing the minimum storage services required to run the benchmark.  If the storage system requires a dedicated network, then the hardware required for that network must be included in the ``storage system``.  If the storage system is hyperconverged, then it will probably share hardware (eg: CPU and/or networking) with the ``host nodes``.
- A **storage scaling unit** is defined as the minimum unit by which the performance and scale of a storage system can be increased. Examples of storage scaling units are “nodes”, “controllers”, “virtual machines” or “shelves”. Benchmark runs with different numbers of storage scaling units allow a reviewer to evaluate how well a given storage solution is able to scale as more scaling units are added.
- A **host node** is defined as the minimum unit by which the load upon the storage system under test can be increased.  Every ``host node`` must run the same number of simulated accelerators.  A ``host node`` can be instantiated by running the MLPerf Storage benchmark code within a Container or within a VM guest image or natively within an entire physical system.  The number of Containers or VM guest images per physical system and the CPU resources per ``host node`` is up to the submitter. Note that the maximum DRAM available to any ``host node`` must be used when calculating the dataset size to be generated for the test. 
- An **ML framework** is a specific version of a software library or set of related libraries for training ML models using a system. Examples include specific versions of Caffe2, MXNet, PaddlePaddle, PyTorch, or TensorFlow.
- A **benchmark** is an abstract problem that can be solved using ML by training a model based on a specific dataset or simulation environment to a target quality level.
- A **reference implementation** is a specific implementation of a benchmark provided by the MLPerf organization.
- A **benchmark implementation** is an implementation of a benchmark in a particular framework by a user under the rules of a specific division.
- A **run** is a complete execution of a benchmark implementation on a system.
- A **benchmark result** is the mean of 5 run results, executed consecutively. The dataset is generated only once for the 5 runs, prior to those runs. The 5 runs must be done on the same machine(s).
- **Nameplate rated power** is defined as the maximum power capacity that can be provided by a power supply unit (PSU), as declared to a certification authority. The nameplate rated power can typically be obtained from the PSU datasheet.
- A **Power Supply Unit (PSU)** is a component which converts an AC or DC voltage input to one or more DC voltage outputs for the purpose of powering a system or subsystem. Power supply units may be redundant and hot swappable.
- **SPEC PTDaemon® Interface (PTDaemon®)** is a software component created by the Standard Performance Evaluation Corporation (SPEC) designed to simplify the measurement of power consumption by abstracting the interface between benchmarking software and supported power analyzers.
- A **Supported power analyzer** is a test device supported by the PTDaemon® software that measures the instantaneous voltage and multiplies it by the instantaneous current, then accumulates these values over a specific time period to provide a cumulative measurement of consumed electrical power. For a listing of supported power analyzers, see https://www.spec.org/power/docs/SPECpower-Device_List.html
- A **System Under Test (SUT)** is the storage system being benchmarked.


- The storage system under test must be described via one of the following **storage system access types**.  The overall solution might support more than one of the below types, but any given benchmark submission must be described by the access type that was actually used during that submission.  An optional vendor-specified qualifier may be specified. This will be displayed in the results table after the storage system access type, for example, “NAS - RDMA”.
  - **Direct-attached media** – any solution using local media on the ``host node``(s); eg: NVMe-attached storage with a local filesystem layered over it.  This will be abbreviated “**Local**” in the results table.
  - **Remotely-attached block device** – any solution using remote block storage; eg: a SAN using FibreChannel, iSCSI, NVMeoF, NVMeoF over RDMA, etc, with a local filesystem implementation layered over it.  This will be abbreviated “**Remote Block**” in the results table.
  - **Shared filesystem using a standards-defined access protocol** – any solution using a version of standard NFS or CIFS/SMB to access storage.  This will be abbreviated “**NAS**” in the results table.
  - **Shared filesystem using a proprietary access protocol** – any network-shared filesystem solution that requires a unique/proprietary protocol implementation to be installed on the ``host node``(s) to access storage; eg: an HPC parallel filesystem.  This will be abbreviated “**Proprietary**” in the results table.
  - **Object** – any solution accessed using an object protocol such as S3,  RADOS, etc.  This will be abbreviated “**Object**” in the results table.
  - **Other** – any solution whose access is not sufficiently described by the above categories.  This will be abbreviated “**Other**” in the results table.

## Performance Metrics

The metrics reported by the benchmark are different for different types of workloads.  They are broken out below.

The benchmark performance metric for Training workloads is **samples per second, subject to a minimum accelerator utilization (AU) defined for that workload**. Higher samples per second is better. 

To pass a benchmark run, the AU should be equal to or greater than the minimum value, and is computed as follows:
```
AU (percentage) = (total_compute_time/total_benchmark_running_time) * 100
```

All the I/O operations from the first **step** are excluded from the AU calculation in order to avoid the disturbance in the averages caused by the startup costs of the data processing pipeline, allowing the AU to more-quickly converge on the steady-state performance of the pipeline.  The I/O operations that are excluded from the AU calculation **are** included in the samples/second reported by the benchmark, however.

If all I/O operations are hidden by compute time, then the `total_compute_time` will equal the `total_benchmark_running_time` and the AU will be 100%.

The total compute time can be derived from the batch size, total dataset size, number of simulated accelerators, and sleep time: 
```
total_compute_time = (records_per_file * total_files) / simulated_accelerators / batch_size * computation_time * epochs.
```

*NOTE: The sleep time has been determined by running the actual MLPerf training workloads including the compute step on real hardware and is dependent on the accelerator type. In this version of the benchmark we include sleep times for **NVIDIA B200 GPUs, as well as AMD MI355 accelerators**. We plan on expanding the measurements to different accelerator types in future releases.*

## Dataset Generation

This section only describes the dataset generation methodology and requirements for Training workloads, the equivalent topic is covered in section 2.2, Checkpointing.

MLPerf Storage uses DLIO to generate synthetic data. Instructions on how to generate the datasets for each benchmark are available [here](https://github.com/mlcommons/storage). The datasets are generated following the sample size distribution and structure of the dataset seeds (see Table 1) for each of the benchmarks. 

**Minimum dataset size**. The MLPerf Storage benchmark script **must be used** to run the benchmarks since it calculates the minimum dataset size for each benchmark.  It does so using the provided number of simulated accelerators and the size of all of the ``host node``’s memory in GB. The minimum dataset size computation is as follows:

- Calculate required minimum samples given number of steps per epoch *(NB:  num_steps_per_epoch is a minimum of 500)*:
```
   min_samples_steps_per_epoch = num_steps_per_epoch * batch_size * num_accelerators_across_all_nodes
```
- Calculate required minimum samples given host memory to eliminate client-side caching effects; *(NB: HOST_MEMORY_MULTIPLIER = 5)*:
```
   min_samples_host_memory_across_all_nodes = number_of_hosts * memory_per_host_in_GB * HOST_MEMORY_MULTIPLIER * 1024 * 1024 * 1024 / record_length
```
- Ensure we meet both constraints:
```
   min_samples = max(min_samples_steps_per_epoch, min_samples_host_memory_across_all_nodes)
```
- Calculate minimum files to generate
```
   min_total_files= min_samples / num_samples_per_file
   min_files_size = min_samples * record_length / 1024 / 1024 / 1024
```

A minimum of ``min_total_files`` files are required which will consume ``min_files_size`` GB of storage.

**Running the benchmark on a subset of a larger dataset**. We support running the benchmark on a subset of the synthetically generated dataset. One can generate a large dataset and then run the benchmark on a subset of that dataset by setting ``num_files_train`` or ``num_files_eval`` smaller than the number of files available in the dataset folder. Note that if the dataset is stored in multiple subfolders, the subset actually used by this run will be evenly selected from all the subfolders. In this case, ``num_subfolders_train`` and ``num_subfolders_eval`` need to be equal to the actual number of subfolders inside the dataset folder in order to generate valid results.

Please note that the log file(s) output during the generation step needs to be included in the benchmark results submission package.

## Single-host Submissions

This section only applies to Training workloads, the equivalent topic is covered in section 2.2.2, "subset mode".

Submitters can add load to the storage system in two orthogonal ways: (1) increase the number of simulated accelerators inside one ``host node`` (i.e., one machine), and/or (2) increase the number of ``host nodes`` connected to the storage system.

For single-host submissions, increase the number of simulated accelerators by changing the ``--num-accelerators`` parameter to the ``benchmark.sh script``. Note that the benchmarking tool requires approximately 0.5GB of host memory per simulated accelerator.

For **single-host submissions**, CLOSED and OPEN division results must include benchmark runs for the maximum simulated accelerators that can be run on ONE HOST NODE, in ONE MLPerf Storage job, without going below the 90% accelerator utilization threshold.

## Multi-host (Distributed) Training Submissions

This setup simulates distributed training of a single training task, spread across multiple ``host nodes``, on a shared dataset. The current version of the benchmark only supports data parallelism, not model parallelism.

Submitters must respect the following for multi-host node submissions:
- All the data must be accessible to all the ``host nodes``. 
- The number of simulated accelerators in each ``host node`` must be identical.

While it is recommended that all ``host nodes`` be as close as possible to identical, that is not required by these Rules.  The fact that distributed training uses a pool-wide common barrier to synchronize the transition from one step to the next of all ``host nodes`` results in the overall performance of the cluster being determined by the slowest ``host node``.

Here are a few practical suggestions on how to leverage a set of non-identical hardware, but these are not requirements of these Rules.  It is possible to leverage very large physical nodes by using multiple Containers or VM guest images per node, each with dedicated affinity to given CPUs cores and where DRAM capacity and NUMA locality have been configured.  Alternatively, larger physical nodes that have higher numbers of cores or additional memory than the others may have those additional cores or memory disabled.

For **distributed training submissions**, CLOSED and OPEN division results must include benchmark runs for the maximum number of simulated accelerators across all ``host nodes`` that can be run in the distributed training setup, without going below the 90% accelerator utilization threshold. Each ``host node`` must run the same number of simulated accelerators for the submission to be valid.

## CLOSED and OPEN Divisions

### CLOSED: virtually all changes are disallowed
CLOSED represents a level playing field where all results are **comparable** across submissions. CLOSED explicitly forfeits flexibility in order to enable easy comparability. 

In order to accomplish that, most of the optimizations and customizations to the AI/ML algorithms and framework that might typically be applied during benchmarking or even during production use must be disallowed.  Optimizations and customizations to the storage system are allowed in CLOSED.

For CLOSED submissions of this benchmark, the MLPerf Storage codebase takes the place of the AI/ML algorithms and framework, and therefore cannot be changed. The sole exception to this rule is if the submitter decides to apply the code change identified in PR#299 of the DLIO repo in github, the resulting codebase will be considered "unchanged" for the purposes of this rule. 

A small number of parameters can be configured in CLOSED submissions; listed in the tables below.

**Table: Training Workload Tunable Parameters for CLOSED**

| Parameter                    | Description                                                                                                                         | Default  |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------|
| *Dataset parameters*         |                                                                                                                                     |          |
| dataset.num_files_train      | Number of files for the training set                                                                                                | --       |
| dataset.num_subfolders_train | Number of subfolders that the training set is stored                                                                                | 0        |
| dataset.data_folder          | The path where dataset is stored                                                                                                    | --       |
|                              |                                                                                                                                     |          |
| *Reader parameters*          |                                                                                                                                     |          |
| reader.read_threads          | Number of threads to load the data                                                                                                  | --       |
| reader.computation_threads   | Number of threads to preprocess the data                                                                          | --       |
| reader.prefetch_size         | An int64 scalar representing the amount of prefetching done, with values of 0, 1, or 2.                                             |          |
| reader.odirect               | Enable ODIRECT mode                                                                                             | False    |
|                              |                                                                                                                                     |          |
| *Checkpoint parameters*      |                                                                                                                                     |          |
| checkpoint.checkpoint_folder | The folder to save the checkpoints                                                                                                  | --       |
|                              |                                                                                                                                     |          |
| *Storage parameters*         |                                                                                                                                     |          |
| storage.storage_root         | The storage root directory                                                                                                          | ./       |
| storage.storage_type         | The storage type                                                                                                                    | local_fs |

**Table: Checkpoint Workload Tunable Parameters for CLOSED**

| Parameter                        | Description                                                 | Default               |
|----------------------------------|-------------------------------------------------------------|-----------------------|
| checkpoint.checkpoint_folder     | The storage directory for writing and reading checkpoints   | ./checkpoints/<model> |
| checkpoint.num_checkpoints_write | The number of checkpoint writes to do in a single dlio call | 10                    |
| checkpoint.num_checkpoints_read  | The number of checkpoint reads to do in a single dlio call  | 10                    |


CLOSED division benchmarks must be referred to using the benchmark name plus the term CLOSED, e.g. “The system was able to support *N ACME X100* accelerators running a CLOSED division 3D U-Net workload at only 8% less than optimal performance.”

### OPEN: changes are allowed but must be disclosed

OPEN allows more **flexibility** to tune and change both the benchmark and the storage system configuration to show off new approaches or new features that will benefit the AI/ML Community. OPEN explicitly forfeits comparability to allow showcasing innovation.

The essence of OPEN division results is that for a given benchmark area, they are “best case” results if optimizations and customizations are allowed.  The submitter has the opportunity to show the performance of the storage system if an arbitrary, but documented, set of changes are made to the data storage environment or algorithms.

Changes to DLIO itself are allowed in OPEN division submissions.  Any changes to DLIO code or command line options must be disclosed. 

While changes to DLIO are allowed, changing the workload itself is not.  Ie: how the workload is processed can be changed, but those changes cannot fundamentally change the purpose and result of the training.  For example, changing the workload imposed upon storage by a training task into a checkpointing task is not allowed.

In addition to what can be changed in the CLOSED submission, the following parameters can be changed in the benchmark.sh script:

| Parameter                    | Description                                | Default                                                             |
|------------------------------|--------------------------------------------|---------------------------------------------------------------------|
| framework                    | The machine learning framework.            | PyTorch |
|                              |                                            |                                                                     |
| *Dataset parameters*         |                                            |                                                                     |
| dataset.format               | Format of the dataset.                     | FLUX.1: ???<br>RetinaNet: ???<br>DLRMv2: ???      |
| dataset.num_samples_per_file |                                            | FLUX.1: ???<br>RetinaNet: ???<br>DLRMv2: ???                      |
|                              |                                            |                                                                     |
| *Reader parameters*          |                                            |                                                                     |
| reader.data_loader           | PyTorch  | PyTorch |


#### OPEN: num_samples_per_file
Changing this parameter is supported only with Tensorflow, using tfrecord datasets. Currently, the benchmark code only supports num_samples_per_file = 1 for Pytorch data loader. To support other values, the data loader needs to be adjusted.

#### OPEN: data_loader
OPEN submissions can have custom data loaders. If a new data loader is added, or an existing data loader is changed, the DLIO code will need to be modified.

#### Execution of OPEN submissions
OPEN division benchmarks must be referred to using the benchmark name plus the term OPEN, e.g. “The system was able to support N ACME X100 accelerators running an OPEN division 3D U-Net workload at only 8% less than optimal performance.”
