# MLPerf Storage Benchmark Suite - Checkpointing Workloads
MLPerf® Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Usage](#usage)
  - [Models](#models)
  - [Benchmark Execution](#benchmark-execution)
  - [Clearing Caches](#clearing-caches)
  - [Metrics and Results Reporting](#metrics-and-results-reporting)
  - [Requirements for Simultaneously Readable and Writable](#requirements-for-simultaneously-readable-and-writable)
  - [OPEN vs CLOSED submissions](#open-vs-closed-submissions)
- [Theory of Operations](#theory-of-operations)
  - [Definitions](#definitions)
  - [Performance Metrics](#performance-metrics)
  - [CLOSED: virtually all changes are disallowed](#closed-virtually-all-changes-are-disallowed)
  - [OPEN: changes are allowed but must be disclosed](#open-changes-are-allowed-but-must-be-disclosed)

---

## Usage

This version of the benchmark does not include offline or online data pre-processing. We are aware that data pre-processing is an important part of the ML data pipeline and we will include it in a future version of the benchmark.

### Checkpointing
#### Models
Benchmark results may be submitted for the following four model configurations. The associated model architectures and parallelism settings are listed below. The number of MPI processes must be set to 8, 64, 512, and 1024 for the respective models for CLOSED submission. 

For CLOSED submissions, participants are not permitted to change the total number of simulated accelerators. However, they may adjust the number of simulated accelerators per host, as long as each host uses more than 4 simulated accelerators. This allows the use of nodes with higher simulated accelerator density and fewer total nodes. Note: the aggregate simulated accelerator memory across all nodes must be sufficient to accommodate the model’s checkpoint size.

**Table 2 LLM models**

| Model                  | 8B     | 70B    | 405B    | 1T     |
|------------------------|--------|--------|---------|--------|
| Hidden dimension       | 4096   | 8192   | 16384   | 25872  |
| FFN size               | 14336  | 28672  | 53248   | 98304  |
| num_attention_heads    | 32     | 128    | 128     | 192    |
| num_kv_heads           | 8      | 8      | 8       | 32     |
| Num layers             | 32     | 80     | 126     | 128    |
| Parallelism (TPxPPxDP) | 1×1×8  | 8×1x8  | 8×32×2  | 8×64×2 |
| Total Processes        | 8      | 64     | 512     | 1024   |
| ZeRO                   | 3      | 3      | 1       | 1      |
| Checkpoint size        | 105 GB | 912 GB | 5.29 TB | 18 TB  |
| Subset: 8-Process Size | 105 GB | 114 GB | 94 GB   | 161 GB |


#### Benchmark Execution
**Checkpoint Modes (global storage vs local storage)** 

There are two operational modes:

* ``default``: Used for shared storage systems. In this mode, the benchmark runs on multiple hosts to write/read the entire checkpoint dataset. The total number of processes (emulated accelerators) must match the number listed in Table 2 (TP×PP×DP = Total Processes).

* ``subset``: Intended for node local storage systems. In this mode, checkpointing is simulated on a single host by writing/reading only a fraction (``num_gpus/TP/PP/DP``) of the checkpoint data, where ``num_gpus`` is the number of simulated accelerators on the host. The only allowed value for number of processes in a subset submission is 8 (the 8B model does not support subset mode as it is already set to 8 processes).

**Checkpoint write and (read) recovery**

For each submission, one must first perform the checkpoint write, then clear the cache if required, and finally perform the checkpoint read. The required command-line flags are:
*Note: Clearing caches is done to ensure that no data for the read phase comes from the filesystem cache*

For a submission, the sequence is the following:
1. Write 10x checkpoints
2. Clear filesystem caches if necessary
3. Read 10x checkpoints

The default options will run the read and write checkpoints in a single mlpstorage call. For example, the following command will execute a sequence of writing 10 checkpoints and reading those same 10 checkpoints.
```bash
mlpstorage checkpointing run --client-host-memory-in-gb 512 --model llama3-8b --num-processes 8 --checkpoint-folder /mnt/checkpoint_test
```

If caches need to be cleared use the following parameters for the WRITE and READ tests. 

* WRITE: ``--num-checkpoints-read=0``
* READ: ``--num-checkpoints-write=0``


In the above example, the write tests would be executed first with this command which will do the writes but no reads.
```bash
mlpstorage checkpointing run --client-host-memory-in-gb 512 --model llama3-8b --num-processes 8 --checkpoint-folder /mnt/checkpoint_test --num-checkpoints-read=0
```

After the write tests complete, clear the caches on your hosts. A standard linux system would use a command like this:
```bash
echo 3 > /proc/sys/vm/drop_caches
```
The end result of "clearing caches" is that 100% data for the read phase should come from the storage system under test and not from the client's filesystem cache. 

Finally, with the same example the read tests would be executed with the following command which indicates no writes during this phase:
```bash
mlpstorage checkpointing run --client-host-memory-in-gb 512 --model llama3-8b --num-processes 8 --checkpoint-folder /mnt/checkpoint_test --num-checkpoints-write=0
```

Caches need to be cleared by the user outside of the mlpstorage tool.

##### Clearing Caches

The checkpoints that are written are quite large. **If the checkpoint size per client node is less than 3x the client node's memory capacity, then the filesystem cache needs to be cleared between the write and read phases.**

Examples:

| Model (Total Size)  | Num Clients & Memory                      | Size for ranks             | Size for 1st and Last Client                             | Need to Clear Caches?                                            |
|---------------------|-------------------------------------------|----------------------------|----------------------------------------------------------|------------------------------------------------------------------|
| Llama3 405b (5.2TB) | 8x (64 Ranks / Node)<br>1024GB per Client | 256x 11.8GB<br>256x 8.85GB | First: 755GB (64x 11.8GB)<br>Last: 566.4GB (64x 8.85GB)  | No (556GB x 3 = 1,699GB which is greater than the client memory) |
| Llama3 70b (912GB)  | 8x (8x Ranks / Node)<br>1024GB per Client | 64x 11.23GB                | First: 89.8GB (8x 14.23GB)<br>Last: Same as First (DP=1) | Yes (89.8 x 3 = 269.5GB which is less than the client memory)    |

In the first case, after 2x checkpoints data that has been written is being flushed from the filesystem cache. This means that after 10x checkpoints a standard Linux system will not have any data in the filesystem cache that would be read for a checkpoint recovery starting back at the first written checkpoint.

In the second case, after 10x checkpoints, 898GB of data will have been written per client with each client having 1024GB of memory. Without clearing caches this data would be read from the filesystem cache

**fsync**

We enforce ``fsync`` to be applied during checkpoint writes to ensure data is flushed to persistent storage. ``fsync`` is enabled by default in all workload configuration files.

**Example Execution Commands**

* ``default`` mode (``WORLD_SIZE = TP*PP*DP`` as listed in Table 2): 
  ```bash
  # Perform checkpoint writes  (make sure the number of hosts is WORLD_SIZE/num_processes_per_host)
  mlpstorage checkpointing run --model llama3-405b \
    --hosts ip1 ip2 .... \
    --num-processes 512 \
    --num-checkpoints-read 0 \
    --checkpoint-folder ./checkpoint_data1 \
    --results-dir ./mlpstorage_results \
    --client-host-memory-in-gb 64

  # Clear the cache (This might require admin access to the system)
  ... 

  # perform checkpoint reads
  mlpstorage checkpointing run --model llama3-405b \
    --hosts ip1 ip2 .... \
    --num-processes 512 \
    --num-checkpoints-write 0 \
    --checkpoint-folder ./checkpoint_data1 \
    --results-dir ./mlpstorage_results \
    --client-host-memory-in-gb 64
  ```
* ``subset`` mode (on a single host with **8 simulated accelerators**)
  ```bash
  # Perform checkpoint writes (data parallelism must match Table 2)
  mlpstorage checkpointing run --model llama3-405b \
    --hosts ip1 \
    --num-processes 8 \
    --num-checkpoints-read 0 \
    --checkpoint-folder ./checkpoint_data1 \
    --results-dir ./mlpstorage_results \
    --client-host-memory-in-gb 64
  # Clear the cache 
  ... 
  # Perform checkpoint read (data parallelism must match Table 2)
  mlpstorage checkpointing run --model llama3-405b \
    --hosts ip1 \
    --num-processes 8 \
    --num-checkpoints-write 0 \
    --checkpoint-folder ./checkpoint_data1 \
    --results-dir ./mlpstorage_results \
    --client-host-memory-in-gb 64
  ```

#### Metrics and Results Reporting
We report the checkpoint time per write / read and I/O throughput from each rank. For each run: 

	* The metric for duration is the maximum time across all processes.
	* The metric for throughput is the minimum across all processes.

A checkpoint workload submission must include 10 checkpoints written and 10 checkpoints read as well as the logs for any optional processes as outlined in section 2.2.5 (clearing caches, storage remapping, etc)

#### Requirements for Simultaneously Readable and Writable

Checkpoint recovery is intended to mimic an environment where a failure has occurred and the data needs to be read by different hosts than wrote the data. 

For storage systems where all hosts can read and write all data simultaneously, the process described above satisfies the requirements.

For storage systems where 1 host has write access to a volume but all hosts have read access, the above process also satisfies the requirements so long as reads can be fulfilled immediately following a write.

For storage systems where 1 host has write access to a volume and a "remapping" process is required for other hosts to read the same data, the time to remap must be measured and included in the submission. 

When a checkpoint is taken/written, it must be written to stable storage, but that checkpoint does not need to be readable by other other hosts yet.  If it is not readable by other hosts immediately after the checkpoint write is complete, if it requires some additional processing or reconfiguration before the checkpoint is readable by other hosts, the time duration between the checkpoint being completed and the earliest time that that checkpoint could be read by a different ``host node`` must be reported in the SystemDescription.yaml file.  That duration between write completion and availability for reading will be added to the time to read/recover from the benchmark.

**Any processes between the write and read phases of checkpointing that are required before data can be read by a different host than wrote the data must be measured and included in the submission. The time for these processes will be added to the recovery time and throughput calculation for submitted scores** 

The system_configuration.yaml document must list whether the solution support simultaneous reads and/or writes as such:
```yaml
System:
  shared_capabilities:
    multi_host_support: True            # False is used for local storage
    simultaneous_write_support: False   # Are simultaneous writes by multiple hosts supported in the submitted configuration
    simultaneous_read__support: True    # Are simultaneous reads by multiple hosts supported in the submitted configuration
```

#### OPEN vs CLOSED submissions
For CLOSED submissions, the total number of processes must be fixed according to Table 2.

For OPEN submissions, the total number of processes may be increased in multiples of (TP×PP) to showcase the scalability of the storage solution.

**Table 3: Configuration parameters and their mutability in CLOSED and OPEN divisions**

| Parameter                          | Meaning                                      | Default value                                 | Changeable in CLOSED | Changeable in OPEN |
|------------------------------------|----------------------------------------------|-----------------------------------------------|----------------------|--------------------|
| --ppn **(USE HOST:SLOTS INSTEAD)**     | Number of processes per node                 | N/A                                           | YES (minimal 4)      | YES (minimal 4)    |
| --num-processes                    | Total number of processes                    | Node local: 8<br>Global: the value in Table 1 | NO                   | YES                |
| --checkpoint-folder                | The folder to save the checkpoint data       | checkpoint/{workload}                         | YES                  | YES                |
| --num-checkpoints-write            | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |
| --num-checkpoints-read             | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |

**The ``--ppn`` syntax above was incorrect for the MPI package the benchmark uses, please use the syntax ``hostname:slotcount`` for the hosts listed in the ``--hosts`` argument.  The ``slotcount`` value has the same meaning as the ``ppn`` value, the number of processes per node to run.**

** By default, --num-checkpoints-read and --num-checkpoints-write are set to be 10. To perform write only, one has to turn off read by explicitly setting ``--num-checkpoints-read=0``; to perform read only, one has to turn off write by explicitly set  ``--num-checkpoints-write=0``

For an OPEN or CLOSED submission, the process must follow:
1. Write 10 checkpoints
2. Clearing Caches or Remapping Volumes if required
3. Read 10 checkpoint

DLIO and mlpstorage both support options to run 10 checkpoints with a single call or run 10 checkpoints as separate invokations of the tools. So long as the process is followed, checkpoints can be executed as a 10 checkpoint batch or individually. 

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

The benchmark performance metrics for Checkpoint workloads (write/take, and read/recover) are **bandwidth while writing, and bandwidth while reading**, plus an additional data point which is the amount of time required, if any, between the completion of writing a checkpoint and the first point at which that checkpoint can be read from a different ``host node``.  That duration between write completeion and availability for reading will be added to the time to read/recover from the benchmark.

**Submitters do not need to use hardware accelerators (e.g., GPUs, TPUs, and other ASICs) when running MLPerf Storage - Checkpointing.**

## CLOSED and OPEN Divisions

### CLOSED: virtually all changes are disallowed
CLOSED represents a level playing field where all results are **comparable** across submissions. CLOSED explicitly forfeits flexibility in order to enable easy comparability. 

In order to accomplish that, most of the optimizations and customizations to the AI/ML algorithms and framework that might typically be applied during benchmarking or even during production use must be disallowed.  Optimizations and customizations to the storage system are allowed in CLOSED.

For CLOSED submissions of this benchmark, the MLPerf Storage codebase takes the place of the AI/ML algorithms and framework, and therefore cannot be changed. The sole exception to this rule is if the submitter decides to apply the code change identified in PR#299 of the DLIO repo in github, the resulting codebase will be considered "unchanged" for the purposes of this rule. 

A small number of parameters can be configured in CLOSED submissions; listed in the tables below.

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

While changes to DLIO are allowed, changing the workload itself is not.  Ie: how the workload is processed can be changed, but those changes cannot fundamentally change the purpose and result of the training.  For example, changing the workload imposed upon storage by a ResNet-50 training task into 3D-Unet training task is not allowed.
