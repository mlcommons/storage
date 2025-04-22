# MLPerf™ Storage V2.0 Benchmark Rules
——————————————————————————————————————————
# MLPerf Storage Benchmark Submission Guidelines v2.0

- [MLPerf Storage Benchmark Submission Guidelines v2.0](#mlperf-storage-benchmark-submission-guidelines-v20)
  - [1. Introduction](#1-introduction)
    - [1.1 Timeline](#11-timeline)
  - [2. Benchmark Overview](#2-benchmark-overview)
    - [2.1 Training](#21-training)
    - [2.2 Checkpointing](#22-checkpointing)
    - [2.3 Vector Database](#23-vector-database)
  - [3 Definitions](#3-definitions)
  - [4. Performance Metrics](#4-performance-metrics)
  - [5. Benchmark Code](#5-benchmark-code)
  - [6. General Rules](#6-general-rules)
    - [6.1. Strive to be fair](#61-strive-to-be-fair)
    - [6.2. System and framework must be available](#62-system-and-framework-must-be-available)
    - [6.3 Non-determinism](#63-non-determinism)
    - [6.4. Result rounding](#64-result-rounding)
    - [6.5. Stable storage must be used](#65-stable-storage-must-be-used)
    - [6.6. Caching](#66-caching)
    - [6.7. Replicability is mandatory](#67-replicability-is-mandatory)
  - [7. Dataset Generation](#7-dataset-generation)
  - [8. Single-host Submissions](#8-single-host-submissions)
  - [9. Distributed Training Submissions](#9-distributed-training-submissions)
  - [10. CLOSED and OPEN Divisions](#10-closed-and-open-divisions)
    - [10.1 CLOSED: virtually all changes are disallowed](#101-closed:-virtually-all-changes-are-disallowed)
    - [10.2 OPEN: changes are allowed but must be disclosed](#102-open:-changes-are-allowed-but-must-be-disclosed)
  - [11. Submission](#11-submission)
    - [11.1 What to submit - CLOSED submissions](#111-what-to-submit---closed-submissions)
    - [11.2 What to submit - OPEN submissions](#112-what-to-submit---open-submissions)
    - [11.3 Directory Structure for CLOSED or OPEN Submissions](#113-directory-structure-for-closed-or-open-submissions)
    - [11.4 System Description](#114-system-description)
      - [11.4.1 System Description JSON](#1141-system-description-json)
      - [11.4.2 System Description PDF](#1142-system-description-pdf)
  - [12. Review](#12-review)
    - [12.1 Visibility of results and code during review](#121-visibility-of-results-and-code-during-review)
    - [12.2 Filing objections](#122-filing-objections)
    - [12.3 Resolving objections](#123-resolving-objections)
    - [12.4 Fixing objections](#124-fixing-objections)
    - [12.5 Withdrawing results / changing division](#125-withdrawing-results-/-changing-division)
  - [13. Roadmap for future MLPerf Storage releases](#13-roadmap-for-future-mlperf-storage-releases)

## 1. Introduction

MLPerf™ Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads. The suite consists of 4 workload categories:

1. Training
2. Checkpointing
3. Model Loading
4. Vector Database

This benchmark attempts to balance two goals. First, we aim for **comparability** between benchmark submissions to enable decision making by the AI/ML Community. Second, we aim for **flexibility** to enable experimentation and to show off unique storage system features that will benefit the AI/ML Community. To that end we have defined two classes of submissions: CLOSED and OPEN. 

The MLPerf name and logo are trademarks of the MLCommons® Association ("MLCommons"). In order to refer to a result using the MLPerf name, the result must conform to the letter and spirit of the rules specified in this document. MLCommons reserves the right to solely determine if a use of its name or logos is acceptable.

### 1.1 Timeline

| Date | Description |
| ---- | ----------- |
| Jun 26, 2024 | Freeze rules & benchmark code. |
| Aug 7, 2024 | Open benchmark for submissions. |
| Aug 21, 2024 | **Submissions due.** |
| Aug 21, 2024 - Sep 11, 2024 | Review period. |
| Sep 11, 2024 | **Benchmark competition results are published.** |


## 2. Benchmark Overview

This version of the benchmark does not include offline or online data pre-processing. We are aware that data pre-processing is an important part of the ML data pipeline and we will include it in a future version of the benchmark.

### 2.1 Training

MLPerf Storage emulates accelerators for the training workloads with the tool DLIO developed by Argonne National Labs. DLIO uses the standard AI frameworks (PyTorch, Tensorflow, Numpy, etc) to load data from storage to memory at the same intensity as a given accelerator.

**This emulation means that submitters do not need to use hardware accelerators (e.g., GPUs, TPUs, and other ASICs) when running MLPerf Storage - Training.**

Instead, our benchmark tool replaces the training on the accelerator for a single batch of data with a ``sleep()`` call. The ``sleep()`` interval depends on the batch size and accelerator type and has been determined through measurement on a system running the real training workload. The rest of the data ingestion pipeline (data loading, caching, checkpointing) is unchanged and runs in the same way as when the actual training is performed.

There are two main advantages to accelerator emulation. First, MLPerf Storage allows testing different storage systems with different types of accelerators. To change the type of accelerator that the benchmark emulates (e.g., to switch to a system with NVIDIA H100 GPUs instead of A100 GPUs), it is enough to adjust the batch size and ``sleep()`` parameter. The second advantage is that MLPerf Storage can put a high load on the storage system simply by increasing the number of emulated accelerators. This allows for testing the behavior of the storage system in large-scale scenarios without purchasing/renting the AI compute infrastructure.

The benchmark suite provides workload [configurations](https://github.com/mlcommons/storage/tree/main/storage-conf/workload) that simulate the I/O patterns of selected workloads listed in Table 1. The I/O patterns for each MLPerf Storage benchmark correspond to the I/O patterns of the MLPerf Training and MLPerf HPC benchmarks (i.e., the I/O generated by our tool for 3D U-Net closely follows the I/O generated by actually running the 3D U-Net training workload). The benchmark suite can also generate synthetic datasets which show the same I/O load as the actual datasets listed in Table 1. 

| Area | Problem | Model | Data Loader | Dataset seed | Minimum AU% |
| ---- | ------- | ----- | ----------- | ------------ | ----------- |
| Vision | Image segmentation (medical) | 3D U-Net | PyTorch | KiTS 19 (140MB/sample) | 90% |
| Vision | Image classification | ResNet-50 | TensorFlow | ImageNet (150KB/sample) | 90% |
| Scientific | Cosmology | parameter prediction | TensorFlow | CosmoFlow N-body simulation (2MB/sample) | 70% |

Table 1: Benchmark description

- Benchmark start point: The dataset is in **shared persistent storage**. 
- Benchmark end point: The measurement ends after a predetermined number of epochs. *Note: data transfers from storage in this test terminate with the data in host DRAM; transfering data into the accelerator memory is not included in this benchmark.*
- Configuration files for the workloads and dataset content can be found [here](https://github.com/mlcommons/storage/tree/main/storage-conf/workload).

### 2.2 Checkpointing

### 2.3 Vector Database

## 3 Definitions 
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
- A **storage system** consists of a defined set of hardware and software resources that provide storage services to one or more ``host nodes``. Storage systems can be hardware based, software-defined, virtualized or cloud based, and must be capable of providing the minimum storage services required to run the benchmark.
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


- The storage system under test must be described via one of the following **storage system access types**.  The overall solution might support more than one of the below types, but any given benchmark submission must be described by the access type that was actually used during that submission.  Specifically, this is reflected in the `system-name.json` file, in the `storage_system→solution_type`, the `storage_system→software_defined` and `storage_system→hyperconverged` fields, and the `networks→protocols` fields.  An optional vendor-specified qualifier may be specified. This will be displayed in the results table after the storage system access type, for example, “NAS - RDMA”.
  - **Direct-attached media** – any solution using local media on the ``host node``(s); eg: NVMe-attached storage with a local filesystem layered over it.  This will be abbreviated “**Local**” in the results table.
  - **Remotely-attached block device** – any solution using remote block storage; eg: a SAN using FibreChannel, iSCSI, NVMeoF, NVMeoF over RDMA, etc, with a local filesystem implementation layered over it.  This will be abbreviated “**Remote Block**” in the results table.
  - **Shared filesystem using a standards-defined access protocol** – any solution using a version of standard NFS or CIFS/SMB to access storage.  This will be abbreviated “**NAS**” in the results table.
  - **Shared filesystem using a proprietary access protocol** – any network-shared filesystem solution that requires a unique/proprietary protocol implementation to be installed on the ``host node``(s) to access storage; eg: an HPC parallel filesystem.  This will be abbreviated “**Proprietary**” in the results table.
  - **Object** – any solution accessed using an object protocol such as S3,  RADOS, etc.  This will be abbreviated “**Object**” in the results table.
  - **Other** – any solution whose access is not sufficiently described by the above categories.  This will be abbreviated “**Other**” in the results table.

## 4. Performance Metrics
TODO: Move to benchmark definitions with metrics for each?

The benchmark performance metric is **samples per second, subject to a minimum accelerator utilization (AU) defined for that workload**. Higher samples per second is better. 

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

*NOTE: The sleep time has been determined by running the actual MLPerf training workloads including the compute step on real hardware and is dependent on the accelerator type. In this version of the benchmark we include sleep times for **NVIDIA A100 and H100 GPUs**. We plan on expanding the measurements to different accelerator types in future releases.*

## 5. Benchmark Code

The MLPerf Storage working group provides a benchmark implementation which includes:
- Scripts to determine the minimum dataset size required for your system, for a given benchmark.
- Scripts for data generation.
- Benchmark tool, based on DLIO, with configuration files for the benchmarks.
- A script for running the benchmark on one host (additional setup is required if you are running a distributed training benchmark – see Section 5). 
- A script for generating the results report (additional scripting and setup may be required if you are running a distributed training benchmark – see Section 5), and potentially additional supporting scripts.

More details on installation and running the benchmark can be found in the [Github repo](https://github.com/mlcommons/storage)

## 6. General Rules
 
The following apply to all results submitted for this benchmark.
### 6.1. Strive to be fair
  Benchmarking should be conducted to measure the framework and storage system performance as fairly as possible. Ethics and reputation matter.
### 6.2. System and framework must be available
- **Available Systems**. If you are measuring the performance of a publicly available and widely-used system or framework, you must use publicly available and widely-used versions of the system or framework. This class of systems will be called Available Systems, and availability here means the system is a publicly available commercial storage system. If you are measuring the performance of a system that is not available at the time of the benchmark results submission, the system must become commercially available **within 6 months** from results publication. Otherwise, the results for that submission will be retracted from the MLCommons results dashboard.
- **RDI Systems**. If you are measuring the performance of an experimental framework or system, you must make the system and framework you use available upon demand for replication by MLCommons. This class of systems will be called RDI (research, development, internal). 

### 6.3 Non-determinism
The data generator in DLIO uses a fixed random seed that must not be changed, to ensure that all submissions are working with the same dataset. Random number generators may be seeded from the following sources:
- Clock
- System source of randomness, e.g. /dev/random or /dev/urandom
- Another random number generator initialized with an allowed seed
Random number generators may be initialized repeatedly in multiple processes or threads. For a single run, the same seed may be shared across multiple processes or threads.

### 6.4. Result rounding
Public results should be rounded normally, to two decimal places.

### 6.5. Stable storage must be used
The MLPerf Storage benchmark will create the dataset on the storage system, in the desired ``dataset format``, before the start of the benchmark run.  The data must reside on stable storage before the actual benchmark testing can run.

### 6.6. Caching
Caching of training data on ``host nodes`` running MLPerf Storage is controlled via a warm up run, dataset size to memory ratios, and changing random seeds between runs.
1. All runs must use a warm-up run before the 5 test runs. 
2. For Training benchmarks, the dataset size must be at least 5x larger than the sum of memory across all of the MLPerf Storage nodes
3. The random seed must change for each run as controlled by the benchmark.py script

### 6.7. Replicability is mandatory
Results that cannot be replicated are not valid results. Replicated results should be within 5% within 5 tries.

## 7. Dataset Generation
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

## 8. Single-host Submissions

Submitters can add load to the storage system in two orthogonal ways: (1) increase the number of simulated accelerators inside one ``host node`` (i.e., one machine), and/or (2) increase the number of ``host nodes`` connected to the storage system.

For single-host submissions, increase the number of simulated accelerators by changing the ``--num-accelerators`` parameter to the ``benchmark.sh script``. Note that the benchmarking tool requires approximately 0.5GB of host memory per simulated accelerator.

For **single-host submissions**, CLOSED and OPEN division results must include benchmark runs for the maximum simulated accelerators that can be run on ONE HOST NODE, in ONE MLPerf Storage job, without going below the 90% accelerator utilization threshold.

## 9. Distributed Training Submissions

This setup simulates distributed training of a single training task, spread across multiple ``host nodes``, on a shared dataset. The current version of the benchmark only supports data parallelism, not model parallelism.

Submitters must respect the following for multi-host node submissions:
- All the data must be accessible to all the ``host nodes``. 
- The checkpoint location must reside in the same storage system that stores the dataset.
- The number of simulated accelerators in each ``host node`` must be identical.

While it is recommended that all ``host nodes`` be as close as possible to identical, that is not required by these Rules.  The fact that distributed training uses a pool-wide common barrier to synchronize the transition from one step to the next of all ``host nodes`` results in the overall performance of the cluster being determined by the slowest ``host node``.

Here are a few practical suggestions on how to leverage a set of non-identical hardware, but these are not requirements of these Rules.  It is possible to leverage very large physical nodes by using multiple Containers or VM guest images per node, each with dedicated affinity to given CPUs cores and where DRAM capacity and NUMA locality have been configured.  Alternatively, larger physical nodes that have higher numbers of cores or additional memory than the others may have those additional cores or memory disabled.

For **distributed training submissions**, CLOSED and OPEN division results must include benchmark runs for the maximum number of simulated accelerators across all ``host nodes`` that can be run in the distributed training setup, without going below the 90% accelerator utilization threshold. Each ``host node`` must run the same number of simulated accelerators for the submission to be valid.

## 10. CLOSED and OPEN Divisions

### 10.1 CLOSED: virtually all changes are disallowed
CLOSED represents a level playing field where all results are **comparable** across submissions. CLOSED explicitly forfeits flexibility in order to enable easy comparability. 

In order to accomplish that, most of the optimizations and customizations to the AI/ML algorithms and framework that might typically be applied during benchmarking or even during production use must be disallowed.  Optimizations and customizations to the storage system are allowed in CLOSED.

For CLOSED submissions of this benchmark, the MLPerf Storage codebase takes the place of the AI/ML algorithms and framework, and therefore cannot be changed. 

A small number of parameters can be configured in CLOSED submissions; listed in the table below.

| Parameter | Description | Default |
| ---- | ---- | ---- |
| *Dataset parameters* | | |
| dataset.num_files_train | Number of files for the training set | -- |
| dataset.num_subfolders_train | Number of subfolders that the training set is stored | 0 |
| dataset.data_folder | The path where dataset is stored | -- |
| *Reader parameters* | | |
| reader.read_threads | Number of threads to load the data | -- |
| reader.computation_threads | Number of threads to preprocess the data(only for bert) | -- |
| reader.transfer_size | An int64 scalar representing the number of bytes in the read buffer. (only supported for Tensorflow) | |
| reader.prefetch_size | An int64 scalar representing the amount of prefetching done, with values of 0, 1, or 2. | |
| *Checkpoint parameters* | | |
| checkpoint.checkpoint_folder | The folder to save the checkpoints | -- |
| *Storage parameters* | | |
| storage.storage_root | The storage root directory | ./ |
| storage.storage_type | The storage type | local_fs |

Table 2: Alterable parameters for CLOSED submissions

CLOSED division benchmarks must be referred to using the benchmark name plus the term CLOSED, e.g. “The system was able to support *N ACME X100* accelerators running a CLOSED division 3D U-Net workload at only 8% less than optimal performance.”

### 10.2 OPEN: changes are allowed but must be disclosed

OPEN allows more **flexibility** to tune and change both the benchmark and the storage system configuration to show off new approaches or new features that will benefit the AI/ML Community. OPEN explicitly forfeits comparability to allow showcasing innovation.

The essence of OPEN division results is that for a given benchmark area, they are “best case” results if optimizations and customizations are allowed.  The submitter has the opportunity to show the performance of the storage system if an arbitrary, but documented, set of changes are made to the data storage environment or algorithms.

Changes to DLIO itself are allowed in OPEN division submissions.  Any changes to DLIO code or command line options must be disclosed. 

While changes to DLIO are allowed, changing the workload itself is not.  Ie: how the workload is processed can be changed, but those changes cannot fundamentally change the purpose and result of the training.  For example, changing the workload imposed upon storage by a ResNet-50 training task into 3D-Unet training task is not allowed.

In addition to what can be changed in the CLOSED submission, the following parameters can be changed in the benchmark.sh script:

| Parameter |  Description | Default |
| --- | --- | --- |
| framework | The machine learning framework. | 3D U-Net: PyTorch; ResNet-50: Tensorflow; Cosmoflow: Tensorflow |
| *Dataset parameters* | | |
| dataset.format | Format of the dataset. | 3D U-Net: .npz; ResNet-50: .tfrecord; Cosmoflow: .tfrecord |
| dataset.num_samples_per_file | Changing this parameter is supported only with Tensorflow, using tfrecord datasets. Currently, the benchmark code only supports num_samples_per_file = 1 for Pytorch data loader. To support other values, the data loader needs to be adjusted. | 3D U-Net: 1; ResNet-50: 1251; Cosmoflow: 1 |
| *Reader parameters* | | |
| reader.data_loader | Supported options: Tensorflow or PyTorch. OPEN submissions can have custom data loaders. If a new data loader is added, or an existing data loader is changed, the DLIO code will need to be modified. | 3D U-Net: PyTorch (Torch Data Loader); ResNet-50: Tensorflow (Tensorflow Data Loader); Cosmoflow: Tensorflow |

**OPEN division benchmark submissions must be run through the benchmark.sh script. The .yaml files cannot be changed (the workload cannot be changed).  The parameters can be changed only via the command line in order to more-explicitly document what was changed.**

OPEN division benchmarks must be referred to using the benchmark name plus the term OPEN, e.g. “The system was able to support N ACME X100 accelerators running an OPEN division 3D U-Net workload at only 8% less than optimal performance.”

## 11. Submission

A **successful run result** consists of a mean samples/second measurement ``(train_throughput_mean_samples_per_second)`` for a complete benchmark run that achieves mean accelerator utilization ``(train_au_mean_percentage)`` equal to or higher than the minimum defined for that workload.

Submissions are made via a ``git push`` into a private MLCommons repository at github.com.  The link to the repo and the required authentication (eg: userid, password) to access that repo will only be given to people who have registered their intent to submit results in this round (see below for the link to the form).

Many ``git push`` operations can be made using that link, but only the last one before the window closes will be considered.  Each ``git push`` operation should include all of the individual result submissions that you want to be included.  Eg: if you want to submit results for A100 and H100, that would be two submissions but only one ``git push`` operation.

Several agreements between the submitter and MLCommons must be completed and signed **before the submission due date** before benchmark results can be submitted.  Note: since these are legal agreements, it can take significant time to get them signed, so please plan ahead.

The [Intention to submit form](https://docs.google.com/forms/d/e/1FAIpQLSdzGRNECkXeRmK4zVH85zRrHh6dvC62V-ZkPuGCfOlMF7j95w/viewform?usp=sharing) is required of everyone who intends to submit results.  We collect the email addresses of submitters so we can contact them if needed, to know how many ``git push`` authentication credentials to create, and to know who to give those credentials to.

Submitters who are not members of MLCommons need to have signed:
- [Non-member Test Agreement](https://drive.google.com/file/d/1rHV0Q_1Rp9pugqG73c6T5Y4Ry1y2bViz/view?usp=sharing)
- [Corporate CLA (Contributor License Agreement)](https://drive.google.com/file/d/1Px6uosXzO8Y5IGvb3bOXD6C0hZEL4_nn/view?usp=drive_link)
- Trademark License Agreement (must be individually requested)

If an organization has already signed these agreements, they do not need to sign them again unless there have been changes to those agreements by MLCommons.  Please look at each document for clarification.

### 11.1 What to submit - CLOSED submissions

A complete submission for one workload (3D-Unet, ResNet, or Cosmoflow) contains 3 folders:
1. **results** folder, containing, for each system:
   - The entire output folder generated by running MLPerf Storage.
   - Final submission JSON summary file ``mlperf_storage_report.json``. The JSON file must be generated using the ``./benchmark.sh reportgen`` script.  The ``./benchmark.sh reportgen`` command must be run on the rank0 machine in order to collect the correct set of files for the submission.
   - Structure the output as shown in [this example](https://github.com/johnugeorge/mlperf-storage-sample-results-v1.0)
   - The logs from the dataset generation step that built the files that this benchmark run read from.
2. **systems** folder, containing:
   - ``<system-name>.json``
   - ``<system-name>.pdf``
   - For system naming examples look [here](https://github.com/mlcommons/storage_results_v0.5/tree/main/closed)
3. **code** folder, containing:
   - Source code of the benchmark implementation. The submission source code and logs must be made available to other submitters for auditing purposes during the review period.

### 11.2 What to submit - OPEN submissions

- Everything that is required for a CLOSED submission, following the same structure.
- Additionally, the source code used for the OPEN Submission benchmark implementations must be available under a license that permits MLCommon to use the implementation for benchmarking.

### 11.3 Directory Structure for CLOSED or OPEN Submissions
```
root_folder (or any name you prefer)
├── Closed
│ 	└──<submitter_org>
│		├── code
│		├── generation_logs
│		├── results
│		│	├──system-name-1
│		│	│	├── unet3d-a100
│		│	│	│	└── ..
│		│	│	├── unet3d-h100
│		│	│	│	└── ..
|		│	|	├── resnet-a100
│		│	│	│	└── ..
|		│	|	├── resnet-h100
│		│	│	│	└── ..
|		│	|	├── cosmoflow-a100	
│		│	│	|	└── ..
|		│	|	└── cosmoflow-h100	
│		│	│		└── ..
│		│	└──system-name-2
│		│	 	├── unet3d-a100
│		│	 	│	└── ..
│		│	 	├── unet3d-h100
│		│	 	│	└── ..
|		│	 	├── resnet-a100
│		│	 	│	└── ..
|		│	 	├── resnet-h100
│		│	 	│	└── ..
|		│	 	├── cosmoflow-a100	
│		│	 	|	└── ..
|		│	 	└── cosmoflow-h100	
│		│	 		└── ..
│		└── systems
│			system-name-1.json
│			system-name-1.pdf
│			system-name-2.json
│			system-name-2.pdf
│
└── Open
 	└──<submitter_org>
		├── code
 		├── generation_logs
 		├── results
 		│	├──system-name-1
 		│	│	├── unet3d-a100
 		│	│	│	└── ..
 		│	│	├── unet3d-h100
 		│	│	│	└── ..
 		│	|	├── resnet-a100
 		│	│	│	└── ..
 		│	|	├── resnet-h100
 		│	│	│	└── ..
 		│	|	├── cosmoflow-a100	
 		│	│	|	└── ..
 		│	|	└── cosmoflow-h100	
 		│	│		└── ..
 		│	└──system-name-2
 		│	 	├── unet3d-a100
 		│	 	│	└── ..
 		│	 	├── unet3d-h100
 		│	 	│	└── ..
 		│	 	├── resnet-a100
 		│	 	│	└── ..
 		│	 	├── resnet-h100
 		│	 	│	└── ..
 		│	 	├── cosmoflow-a100	
 		│	 	|	└── ..
 		│	 	└── cosmoflow-h100	
 		│	 		└── ..
		└── systems
			system-name-1.json
			system-name-1.pdf
			system-name-2.json
			system-name-2.pdf
```

### 11.4 System Description

The purpose of the system description is to provide sufficient detail on the storage system under test, and the ``host nodes`` running the test, plus the network connecting them, to enable full reproduction of the benchmark results by a third party. 

Each submission must contain a ``<system-name>.json`` file and a ``<system-name>.pdf`` file.  If you submit more than one benchmark result, each submission must have a unique ``<system-name>.json`` file and a ``<system-name>.pdf`` file that documents the system under test and the environment that generated that result, including any configuration options in effect.

Note that, during the review period, submitters may be asked to include additional details in the JSON and pdf to enable reproducibility by a third party.

#### 11.4.1 System Description JSON

The ``<system-name>.json`` file must be pass a validation check with the JSON schema in use for V1.0.  The [schema and two examples of it being used](https://drive.google.com/drive/folders/1ZXNUXN4L8amD0Ba38GoIOt761b73F8bs) are provided.  For example, ``check-jsonschema`` is a convenient tool that is present in many linux distributions, but other tools may be used.

#### 11.4.2 System Description PDF

The goal of the pdf is to complement the JSON file, providing additional detail on the system to enable full reproduction by a third party. We encourage submitters to add details that are more easily captured by diagrams and text description, rather than a JSON.

This file is should include everything that a third party would need in order to recreate the results in the submission, including product model numbers or hardware config details, unit counts of drives and/or components, system and network topologies, software used with version numbers, and any non-default configuration options used by any of the above.

A great example of a system description pdf can be found [here](https://github.com/mlcommons/storage_results_v0.5/tree/main/closed/DDN/systems).


**Cover page**

The following information is required to be included in the system description PDF:

- System name of the submission
- Submitter name
- Submission date
- Version of the benchmark
- Solution type of the submission
- Submission division (OPEN or CLOSED)

**Mandatory Power requirements**

Systems that require customer provisioning of power (for example, systems intended to be deployed in on-premises data centers or in co-located data centers) shall include a “Power Requirements Table”. Systems designed to only run in a cloud or hyper-converged environment do not have to include this table.

The power requirements table shall list all hardware devices required to operate the storage system. Shared network equipment also used for client network communication and optional storage management systems do not need to be included. The power requirements table shall include:

1. Every component in the system that requires electrical power.
2. For each component, every PSU for each system component.
3. For each PSU, the PSU nameplate rated power.
4. For each PSU (or redundant groups of PSUs0, the design power.

Two examples of a power requirements tables are shown below:

**Power Requirements Table** (Large system example)

| System component     | Power supply unit | Nameplate rated power | Design power   |
| -------------------- | ----------------- | --------------------- | -------------- |
| Storage controller 1 | Power supply 1    | 1200 watts            | 3600 watts     |
|                      | Power supply 2    | 1200 watts            |                |
|                      | Power supply 3    | 1200 watts            |                |
|                      | Power supply 4    | 1200 watts            |                |
| Storage shelf 1      | Power supply 1    | 1000 watts            | 1000 watts     |
|                      | Power supply 2    | 1000 watts            |                |
| Network switch 1     | Power supply 1    | 1200 watts            | 1200 watts     |
|                      | Power supply 2    | 1200 watts            |                |
| **Totals**           |                   | **9200 watts**        | **5800 watts** |

**Power Requirements Table** (Direct-attached media system example)

| System component     | Power supply unit | Nameplate rated power | Design power   |
| -------------------- | ----------------- | --------------------- | -------------- |
| NVMe SSD 1           | 12VDC supply      | 10 watts              | 10 watts       |
|                      | 3.3VDC supply     | 2 watts               | 2 watts        |
| **Totals**           |                   | **12 watts**          | **12 watts**   |

System component and power supply unit names in the above tables are examples. Consistent names should be used in bill-of-material documentation, system diagrams and descriptive text.

**Optional information**

The following *recommended* structure of systems.pdf provides a starting point for additional optional information. Submitters are free to adjust this structure as they see fit.

If the submission is for a commercial system, a pdf of the product spec document can add significant value.  If it is a system that does not have a spec document (e.g., a research system, HPC etc), or the product spec pdf doesn’t include all the required detail, the document can contain (all these are optional):

- Recommended: High-level system diagram e.g., showing the ``host node``(s), storage system main components, and network topology used when connecting everything (e.g., spine-and-leaf, butterfly, etc.), and any non-default configuration options that were set during the benchmark run.
- Optional: Additional text description of the system, if the information is not captured in the JSON, e.g., the storage system’s components (make and model, optional features, capabilities, etc) and all configuration settings that are relevant to ML/AI benchmarks.  If the make/model doesn’t specify all the components of the hardware platform it is running on, eg: it’s an Software-Defined-Storage product, then those should be included here (just like the client component list).
- Optional: physical requirements – If the system requires the physical deployment of hardware, consider including the number of rack units, required supporting equipment, and any physical constraints on how the equipment must be installed into an industry-standard rack, such as required spacing, weight constraints, etc. We recommended the following three categories for the text description:
  1. Software, 
  2. Hardware, and
  3. Settings.

## 12. Review

### 12.1 Visibility of results and code during review

During the review process, only certain groups are allowed to inspect results and code.
| Group | Can Inspect |
| --- | --- |
| Review committee | All results, all code |
| Submitters | All results, all code |
| Public | No results, no code |

### 12.2 Filing objections

Submitters must officially file objections to other submitter’s code by creating a GitHub issue prior to the “Filing objections” deadline that cites the offending lines, the rules section violated, and, if pertinent, corresponding lines of the reference implementation that are not equivalent. Each submitter must file objections with a “by <org>” tag and a “against <org>” tag. Multiple organizations may append their “by <org>” to an existing objection if desired. If an objector comes to believe the objection is in error they may remove their “by <org>” tag. All objections with no “by <org>” tags at the end of the filing deadline will be closed. Submitters should file an objection, then discuss with the submitter to verify if the objection is correct. Following filing of an issue but before resolution, both objecting submitter and owning submitter may add comments to help the review committee understand the problem. If the owning submitter acknowledges the problem, they may append the “fix_required” tag and begin to fix the issue.

### 12.3 Resolving objections

The review committee will review each objection, and either establish consensus or vote. If the committee votes to support an objection, it will provide some basic guidance on an acceptable fix and append the “fix_required” tag. If the committee votes against an objection, it will close the issue.

### 12.4 Fixing objections

Code should be updated via a pull request prior to the “fixing objections” deadline. Following submission of all fixes, the objecting submitter should confirm that the objection has been addressed with the objector(s) and ask them to remove their “by <org> tags. If the objector is not satisfied by the fix, then the review committee will decide the issue at its final review meeting. The review committee may vote to accept a fix and close the issue, or reject a fix and request the submission be moved to open or withdrawn.

### 12.5 Withdrawing results / changing division

Anytime up until the final human readable deadline (typically within 2-3 business days before the press call, so June 5th, 2024, in this case), an entry may be withdrawn by amending the pull request.  Alternatively, an entry may be voluntarily moved from the closed division to the open division.  Each benchmark results submission is treated separately for reporting in the results table and in terms of withdrawing it.  For example, submitting a 3D-Unet run with 20 clients and 80 A100 accelerators is separate from submitting a 3D-Unet run with 19 clients and 76 accelerators.

## 13. Roadmap for future MLPerf Storage releases

The Working Group is very interested in your feedback. Please contact storage-chairs@mlcommons.org with any suggestions.

Our working group aims to add the following features in a future version of the benchmark:
- We plan to add support for the “data pre-processing” phase of AI/ML workload as we are aware that this is a significant load on a storage system and is not well represented by existing AI/ML benchmarks. 
- Add support for other types of storage systems (e.g., Object Stores) in the CLOSED division.
- Expand the number of workloads in the benchmark suite e.g.,add a large language model (GPT3), and a diffusion model (Stable Diffusion).
- Add support for PyTorch and Tensorflow in the CLOSED division for all workloads.
- Continue adding support for more types of accelerators.
- We plan to add support for benchmarking a storage system while running more than one MLPerf Storage benchmark at the same time (ie: more than one Training job type, such as 3DUnet and Recommender at the same time), but the current version requires that a submission only include one such job type per submission. 
