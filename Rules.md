# MLPerf™ Storage V2.0 Benchmark Validation Rules
——————————————————————————————————————————

- [MLPerf Storage V2.0 Benchmark Validation Rules](#mlperf-storage-v20-benchmark-validation-rules)
  - [1. Introduction](#1-introduction)
  - [2. Core/Common Rules](#2-core-common-rules)
    - [2.1. Core/Common POSIX API Rules](#21-ccore-common-posix-api-rules)
    - [2.2. Core/Common Object API Rules](#22-ccore-common-object-api-rules)
  - [3. Validating the Training Options](#3-validating-the-training-options)
    - [3.1. Training Sizing Options](#31-training-sizing-options)
    - [3.2. Training Generation Options](#32-training-ganeration-options)
    - [3.3. Training Run Options](#33-training-run-options)
    - [3.4. Training Access Via POSIX API Options](#34-training-access-via-posix-api-options)
    - [3.5. Training Access Via Object API Options](#35-training-access-via-object-API-options)
    - [3.6. Training OPEN versus CLOSED Options](#36-training-open-versus-closed-options)
  - [4. Validating the Checkpointing Options](#4-validating-the-checkpointing-options)
    - [4.1. Checkpointing Sizing Options](#41-checkpointing-sizing-options)
    - [4.2. Checkpointing Generation Options](#42-checkpointing-generation-options)
    - [4.3. Checkpointing Run Options](#43-checkpointing-run-options)
    - [4.4. Checkpointing Access Via POSIX API Options](#44-checkpointing-access-via-posix-api-options)
    - [4.5. Checkpointing Access Via Object API Options](#45-checkpointing-access-via-object-API-options)
    - [4.6. Checkpointing OPEN versus CLOSED Options](#46-checkpointing-open-versus-closed-options)
    - [4.7. Storage System Must Be Simultaneously R/W or Remappable](#47-storage-system-must-be-simultaneously-rw-or-remappable)
  - [5. Validating the VDB Options](#5-validating-the-vdb-options)
    - [5.1. VDB Sizing Options](#51-vdb-sizing-options)
    - [5.2. VDB Generation Options](#52-vdb-generation-options)
    - [5.3. VDB Run Options](#53-vdb-run-options)
    - [5.4. VDB Access Via POSIX API Options](#54-vdb-access-via-posix-api-options)
    - [5.5. VDB Via Object API Options](#55-vdb-access-via-object-API-options)
    - [5.6. VDB OPEN versus CLOSED Options](#56-vdb-open-versus-closed-options)
  - [6. Validating the KVCache Options](#6-validating-the-kvcache-options)
    - [6.1. KVCache Sizing Options](#61-kvcache-sizing-options)
    - [6.2. KVCache Generation Options](#62-kvcache-generation-options)
    - [6.3. KVCache Run Options](#63-kvcache-run-options)
    - [6.4. KVCache Access Via POSIX API Options](#64-kvcache-access-via-posix-api-options)
    - [6.5. KVCache Access Via Object API Options](#65-kvcache-access-via-object-API-options)
    - [6.6. KVCache OPEN versus CLOSED Options](#66-kvcache-open-versus-closed-options)

# 1.  Introduction

These are the requirements for the *submission validation checker* for version 2.0 of the MLPerf™ Storage benchmark,
but since the `mlpstorage` tool will be responsible for generating the vast majority (if not all) of the contents of a submission, it is also a spec for what `mlpstorage` should generate.

The *submission validation checker* should check that the tested directory hierarachy matches the below requirements and output messages for all cases where it does not match.
The tool should make it's best effort to continue testing all the other aspects of the directory hierarchy after any given failure.
If the tested directory hierarchy does not meet all of the below requirements, then it should be labelled as invalid and the validation check should fail.

Even if the structure of a submission package matches the spec, the options that were used to run the benchmark may not fall within acceptable bounds,
so we need the *submission validation checker* to check for illegal/inapproriate option settings,
and for semantic mismatches between different options that were used.

The `mlpstorage` tool must be used to run the benchmarks, submitters are not allowed to run the underlying tools (eg: DLIO) directly to generate a submission package.

1.1. **mlpstorageGeneratesHierarchy** -- The `mlpstorage` command must obtain (somehow) the pathname of the output file directory hierarchy and directly create and/or append to the files within that hierarchy to successively build out the submission folder.  We don't want the submitter to manually create anything in that hierarchy except for the SystemDescription.* files (if we can help it).

# 2.  Core/Common Rules for All Submissions

## 2.1.  Core/Common POSIX API Rules

2.1.1. **submitterRootDirectory** --  The submission structure must start from a single directory whose name is the name of the submitter.  This can be any string, but a blank or any other character in that string that cannot be part of a POSIX filename should be replaced 1-for-1 with a dash character.

2.1.2. **topLevelSubdirectories** --  Within the top-level directory of the submission structure there must be a directory named "closed" and/or one named "open", and nothing more.  These names are case-sensitive.

2.1.3. **openMatchesClosed** --  The "open" directory hierarchy should be constructed identically to the "closed" directory hierarchy describe just below.

2.1.4. **closedSubmitterDirectory** --  Within the "closed" directory there must be a single directory whose name is the name of the submitter (the same as the top-level directory).

2.1.5. **requiredSubdirectories** --  Within the submitter directory mentioned just above, there must be exactly three directories: "code", "results", and "systems".  These names are case-sensitive.

2.1.6. **codeDirectoryContents** --  The "code" directory must include a complete copy of the MLPerf Storage github repo that was used to run the test that resulted in the "results" directory's contents.
If this is in the "open" hierarchy, any modifications made to the benchmark code must be included here, and if this is in the "closed" hierarchy, there must be no changes to the benchmark code.
Note that in both cases this must be the code that was actually run to generate those results.  In a CLOSED submission, the *submission validator* should do an md5sum of the code directory hierarchy, compare that to a value hard-coded into the validator code, and fail the validation if there is a difference.

2.1.7. **systemsDirectoryFiles** --  The "systems" directory must contain two files for each "system name", a .yaml file and a .pdf file, and nothing more.  Each of those files must be named with the "system name".
Eg: for a system-under-test named "Big_and_Fast_4000_buffered", there must be a "Big_and_Fast_4000_buffered.yaml" and a "Big_and_Fast_4000_buffered.pdf" file.  These names are case-sensitive.

2.1.8. **resultsDirectorySystems** --  The "results" directory, whether it is within the "closed' or "open" hierarchies, must include one or more directories that are the names of the systems-under-test.  Eg: a system name could be "Big_and_Fast_4000_buffered".
This name can be anything the submitter wants, it is just a name to both idenfity the set of results that were collected from a given	
configuration of storage system and to link together those results with the .pdf and .yaml files that describe the system-under-test.

2.1.9. **identicalSystemConfig** --  All the configuration parameters and hardware and software components of the system-under-test that are part of a given *system name* must be identical.  Any changes to those configuration parameters or hardware or software must be submitted as a separate *system name*, so we should compare the configuration parameters and hardware and software components to verify that they're the same across all the tests and runs within the given *system name* directory hierarchy, to the extent that we can.  The *system names*  are case-sensitive.

2.1.10. **workloadCategories** --  Within a *system name* directory in the "results" directory, there must be one or both of the following directories, and nothing else: "training", and/or "checkpointing".  These names are case-sensitive.

2.1.11. **trainingWorkloads** --  Within the "training" directory, there must be one or more of the following *workload directories*, and nothing else: "unet3d", "resnet50" and/or "cosmoflow".  These names are case-sensitive.

2.1.12. **trainingPhases** --  Within the *workload directories* in the "training" hierarchy, there must exist *phase directories* named "datagen" and "run", and nothing else.  These names are case-sensitive.

2.1.13. **datagenTimestamp** --  Within the "datagen" *phase directory* within the "training" directory hierarchy, there must be exactly one *timestamp directory* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

2.1.14. **datagenFiles** --  Within the *timestamp directory* within the "datagen" *phase*, there must exist the following files: "training_datagen.stdout.log", "training_datagen.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.15. **datagenDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory*  must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.16. **runResultsJson** --  Within the "run" *phase directory* within the "training" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

2.1.17. **runTimestamps** --  Within the "run" *phase directory* within the "training" directory hierarchy, there must also be exactly 6 subdirectories named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.  Note that the 1st of those 6 is the *warm up* run and will not be included in the reported performance.

2.1.18. **runTimestampGap** --  The timestamp (the day and time) represented by the name of each *timestamp directory* must be separated by less than the duration of a single *timestamp directory* from it's neighboring *timestamp directories*.  Ie: the gap between a consecutive pair of *timestamp directories* must be short enough that we can be sure that there was no benchmark activity between them.

2.1.19. **runFiles** --  Within each *timestamp directory* within the "run" *phase*, there must exist the following files: "training_run.stdout.log", "training_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.20. **runDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.21. **checkpointingWorkloads** --  Within the "checkpointing" directory, there must be one or more of the following *workload directories*, and nothing else: "llama3-8b", "llama3-70b", "llama3-405b", and/or "llama3-1t".  These names are case-sensitive.

2.1.22. **checkpointingResultsJson** --  Within the *workload directories* within the "checkpointing" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

2.1.23. **checkpointingTimestamps** --  Within the *workload directories* within the "checkpointing" directory hierarchy, there must also be exactly ten *timestamp directories* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

2.1.24. **checkpointingTimestampGap** --  The timestamp (the day and time) represented by the name of each *timestamp directory* must be separated by less than the duration of a single *timestamp directory* from it's neighboring *timestamp directories*.  Ie: the gap between a consecutive pair of *timestamp directories* must be short enough that we can be sure that there was no benchmark activity between them.

2.1.25. **checkpointingFiles** --  Within the *timestamp directories* within the "checkpointing" directory hierarchy, there must exist the following files: "checkpointing_run.stdout.log", "checkpointing_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.26. **checkpointingDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.27. **directoryDiagram** --  Pictorially, here is what this looks like:
```
root_folder (or any name you prefer)
├── Closed
│ 	└──<submitter_org>
│	  	├── code
│	  	├── results
│	  	│	└──system-name-1
│	  	│	 	├── training
│	  	│	 	│	├── unet3d
│	  	│		│	│	├── datagen
│	  	│		│	│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│	│		└── dlio_config
│	  	│		│	│	└── run
│	  	│		│	│		├──results.json
│	  	│		│	│		├── YYYYMMDD_HHmmss
│	  	│		│	│		│	└── dlio_config 
│	  	│		│	│		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	│		└── YYYYMMDD_HHmmss
│	  	│		│	│			└── dlio_config
│	  	│	 	│	├── resnet50
│	  	│		│	│	├── datagen
│	  	│		│	│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│	│		└── dlio_config
│	  	│		│	│	└── run
│	  	│		│	│		├──results.json
│	  	│		│	│		├── YYYYMMDD_HHmmss
│	  	│		│	│		│	└── dlio_config 
│	  	│		│	│		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	│		└── YYYYMMDD_HHmmss
│	  	│		│	│			└── dlio_config
│	  	│	 	│	└── cosmoflow
│	  	│		│	 	├── datagen
│	  	│		│	 	│	└── YYYYMMDD_HHmmss
│	  	│		│	 	│		└── dlio_config
│	  	│		│	 	└── run
│	  	│		│			├──results.json
│	  	│		│	 		├── YYYYMMDD_HHmmss
│	  	│		│	 		│	└── dlio_config 
│	  	│		│	 		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	 		└── YYYYMMDD_HHmmss
│	  	│		│	 			└── dlio_config
│	  	│	 	└── checkpointing
│	  	│	 		├── llama3-8b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		├── llama3-70b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		├── llama3-405b
│	  	│			│	├──results.json
│	  	│			│	├── YYYYMMDD_HHmmss
│	  	│			│	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│			│	└── YYYYMMDD_HHmmss
│	  	│			│		└── dlio_config
│	  	│	 		└── llama3-1t
│	  	│				├──results.json
│	  	│			 	├── YYYYMMDD_HHmmss
│	  	│			 	│	└── dlio_config 
│	  	│			 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│				└── YYYYMMDD_HHmmss
│	  	│			 		└── dlio_config
│	  	└── systems
│	  		├──system-name-1.yaml
│	  		├──system-name-1.pdf
│	  		├──system-name-2.yaml
│	  		└──system-name-2.pdf
│
└── Open
 	└──<submitter_org>
		├── code
		├── results
		│	└──system-name-1
		│	 	├── training
		│	 	│	├── unet3d
		│		│	│	├── datagen
		│		│	│	│	└── YYYYMMDD_HHmmss
		│		│	│	│		└── dlio_config
		│		│	│	└── run
		│		│	|		├──results.json
		│		│	│		├── YYYYMMDD_HHmmss
		│		│	│		│	└── dlio_config 
		│		│	│		... (5x Runs per Emulated Accelerator Type)
		│		│	│		└── YYYYMMDD_HHmmss
		│		│	│			└── dlio_config
		│	 	│	├── resnet50
		│		│	│	├── datagen
		│		│	│	│	└── YYYYMMDD_HHmmss
		│		│	│	│		└── dlio_config
		│		│	│	└── run
		│		│	|		├──results.json
		│		│	│		├── YYYYMMDD_HHmmss
		│		│	│		│	└── dlio_config 
		│		│	│		... (5x Runs per Emulated Accelerator Type)
		│		│	│		└── YYYYMMDD_HHmmss
		│		│	│			└── dlio_config
		│	 	│	└── cosmoflow
		│		│	 	├── datagen
		│		│	 	│	└── YYYYMMDD_HHmmss
		│		│	 	│		└── dlio_config
		│		│	 	└── run
		│		│			├──results.json
		│		│	 		├── YYYYMMDD_HHmmss
		│		│	 		│	└── dlio_config 
		│		│	 		... (5x Runs per Emulated Accelerator Type)
		│		│	 		└── YYYYMMDD_HHmmss
		│		│	 			└── dlio_config
		│	 	└── checkpointing
		│	 		├── llama3-8b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		├── llama3-70b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		├── llama3-405b
		│			|	├──results.json
		│			│	├── YYYYMMDD_HHmmss
		│			│	│	└── dlio_config 
		│			│	... (10x Runs for Read and Write. May be combined in a single run)
		│			│	└── YYYYMMDD_HHmmss
		│			│		└── dlio_config
		│	 		└── llama3-1t
		│				├──results.json
		│			 	├── YYYYMMDD_HHmmss
		│			 	│	└── dlio_config 
		│				... (10x Runs for Read and Write. May be combined in a single run)
		│				└── YYYYMMDD_HHmmss
		│			 		└── dlio_config
		└── systems
			├──system-name-1.yaml
			├──system-name-1.pdf
			├──system-name-2.yaml
			└──system-name-2.pdf
```
2.29. **dlioLog** --  Since the "dlio_log" subdirectory has a similar structure in all cases, it is describe pictorially just below:
```
└── YYYYMMDD_HHmmss
    ├── [training|checkpointing]_[datagen|run].stdout.log
    ├── [training|checkpointing]_[datagen|run].stderr.log
    ├── *[output|per_epoch_stats|summary].json
    ├── dlio.log
    └── dlio_config
        ├── config.yaml
        ├── hydra.yaml
        └── overrides.yaml
```

## 2.2.  Core/Common Object API Rules

# 3.  Validating the Training Workloads

## 3.1.  Training Sizing Options

3.1.1. **trainingVerifyDatasizeUsage** -- The *submission validator* must verify that the *datasize* option was used by finding the entry(s) in the log file showing its use.

3.1.2. **trainingRecalculateDatasetSize** -- The *submission validator* must recalculate the minimum dataset size by using the provided number of simulated accelerators and the sizes of all of the host node’s memory as reported in the logfiles as described below and fail the run if the size recorded in the run's logfile doesn't exactly match the recalculated value.
  * Calculate required minimum samples given number of steps per epoch (NB: `num_steps_per_epoch` is a minimum of 500):
     * `min_samples_steps_per_epoch = num_steps_per_epoch * batch_size * num_accelerators_across_all_nodes`
  * Calculate required minimum samples given host memory to eliminate client-side caching effects; (NB: HOST_MEMORY_MULTIPLIER = 5):
     * `min_samples_host_memory_across_all_nodes = number_of_hosts * memory_per_host_in_GB * HOST_MEMORY_MULTIPLIER * 1024 * 1024 * 1024 / record_length`
  * Ensure we meet both constraints:
     * `min_samples = max(min_samples_steps_per_epoch, min_samples_host_memory_across_all_nodes)`
  * Calculate minimum files to generate
     * `min_total_files= min_samples / num_samples_per_file`
     * `min_files_size = min_samples * record_length / 1024 / 1024 / 1024`
  * A minimum of `min_total_files` files are required which will consume `min_files_size` GB of storage.

## 3.2.  Training Generation Options

3.2.1. **trainingDatagenMinimumSize** --  The amount of data generated during the *datagen* phase must be equal **or larger** -- than the amount of data calculated during the *datasize* phase or the run must be failed.

## 3.3.  Training Run Options

3.3.1. **trainingRunDataMatchesDatasize** -- The amount of data the *run* phase is told to use must be exactly equal to the *datasize* value calculated earlier, but can be less than the value used in the *datagen* phase.  To express that, you can run the benchmark on a subset of that dataset by setting `num_files_train` or `num_files_eval` smaller than the number of files available in the dataset folder, but `num_subfolders_train` and `num_subfolders_eval` must be to be equal to the actual number of subfolders inside the dataset folder in order to generate valid results.

3.3.2. **trainingAcceleratorUtilizationCheck** -- To pass a benchmark run, the AU (Accelerator Utilization) should be equal to or greater than the minimum value:
  * `total_compute_time = (records_per_file * total_files) / simulated_accelerators / batch_size * computation_time * epochs`
  * `AU = (total_compute_time/total_benchmark_running_time) * 100`
  * All the I/O operations from the first step are excluded from the AU calculation. The I/O operations that are excluded from the AU calculation are included in the samples/second reported by the benchmark, however.

3.3.3. **trainingSingleHostSimulatedAccelerators** -- For single-host submissions, increase the number of simulated accelerators by changing the `--num-accelerators` parameter to the benchmark.sh script. Note that the benchmarking tool requires approximately 0.5GB of host memory per simulated accelerator.

3.3.4. **trainingSingleHostClientLimit** -- For single-host submissions, in both CLOSED and OPEN division results, the validator should fail the run if there is more than one client node used during that run.

3.3.5. **trainingDistributedDataAccessibility** -- For distributed Training submissions, all the data must be accessible to all the host nodes.  **_(not clear how to check this, so maybe remove?)_**

3.3.6. **trainingIdenticalAcceleratorsPerNode** -- For distributed Training submissions, the number of simulated accelerators in each host node must be identical.

3.3.7. **trainingNodeCapabilityConsistency** -- For distributed Training submissions, the *submission validation checker* should emit a warning (not fail the validation) if the physical nodes that run the benchmark code are widely enough different in their capability.  **_(not clear we should do this, so maybe remove?)_**

## 3.4.  Training Access Via POSIX API Options

3.4.1. **trainingMlpstoragePathArgs** --  The arguments to `mlpstorage` that set the directory pathname where the dataset is stored and the directory where the output logfiles are stored must both be set and must be set to different values.

3.4.2. **trainingMlpstorageFilesystemCheck** --  The `mlpstorage` command should do a "df" command on the directory pathname where the dataset is stored and another one on the directory pathname where the output logfiles are stored and record those values in the logfile.  The *submission validator* should find those entries in the run's logfile and verify that they are different filesystems.  We don't want the submitter to, by acccident, place the logfiles onto the storage system under test since that would skew the results.

## 3.5.  Training Access Via Object API Options

## 3.6.  Training OPEN versus CLOSED Options

3.6.1. **trainingClosedSubmissionChecksum** -- For CLOSED submissions of this benchmark, the MLPerf Storage codebase cannot be changed, so the *submission validation checker* SHOULD do an `md5sum` of the code directory hierachy in the submission package and verify that that matches a precalculated checksum stored as a literal in the validator's codebase.

3.6.2. **trainingClosedSubmissionParameters** -- For CLOSED submissions of this benchmark, only a small number of parameters can be modified, and those parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

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
| reader.computation_threads   | Number of threads to preprocess the data (only for resnet)                                                                          | --       |
| reader.transfer_size         | An int64 scalar representing the number of bytes in the read buffer. (only supported for Tensorflow models -- Resnet and Cosmoflow) |          |
| reader.prefetch_size         | An int64 scalar representing the amount of prefetching done, with values of 0, 1, or 2.                                             |          |
| reader.odirect               | Enable ODIRECT mode for Unet3D Training                                                                                             | False    |
|                              |                                                                                                                                     |          |
| *Storage parameters*         |                                                                                                                                     |          |
| storage.storage_root         | The storage root directory                                                                                                          | ./       |
| storage.storage_type         | The storage type                                                                                                                    | local_fs |

3.6.3. **trainingOpenSubmissionParameters** -- For OPEN submissions of this benchmark, only a few additional parameters can be modified over those allowed in CLOSED, and those additional parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

**Table: Training Workload Tunable Parameters for OPEN**

| Parameter                    | Description                                | Default                                                                               |
|------------------------------|--------------------------------------------|---------------------------------------------------------------------------------------|
| framework                    | The machine learning framework.            | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow                   |
|                              |                                            |                                                                                       |
| *Dataset parameters*         |                                            |                                                                                       |
| dataset.format               | Format of the dataset.                     | 3D U-Net: .npz<br>ResNet-50: .tfrecord<br>Cosmoflow: .tfrecord                        |
| dataset.num_samples_per_file |                                            | 3D U-Net: 1<br>ResNet-50: 1251<br>Cosmoflow: 1                                        |
|                              |                                            |                                                                                       |
| *Reader parameters*          |                                            |                                                                                       |
| reader.data_loader           | Supported options: Tensorflow or PyTorch.  | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow                   |

# 4.  Validating the Checkpointing Workloads

## 4.1.  Checkpointing Sizing Options

## 4.2.  Checkpointing Generation Options

## 4.3.  Checkpointing Run Options

4.3.1. **checkpointDataSizeRatio** -- The checkpoint data written per client node must be more than 3x the client node's memory capacity, otherwise the filesystem cache needs to be cleared between the write and read phases.

4.3.2. **checkpointFsyncVerification** -- We must verify that all the benchmark workload configuration files have been set to do an fsync call at the end of each of the 10 checkpoint writes.

4.3.3. **checkpointModelConfigurationReq** -- The benchmark must be run with one of the four model configuration detailed below.

4.3.4. **checkpointAggregateAcceleratorMemory** -- The aggregate simulated accelerator memory across all nodes must be sufficient to accommodate the model’s checkpoint size.  That is, the GB of memory associated with the chosen accelerator (eg: H100) times the accelerator count must be equal to or greater than the total checkpoint size for that scale of checkpoint.  (see table 2)

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

4.3.5. **checkpointSubsetRunValidation** --  The `mlpstorage` command must accept a parameter telling it that this is a *subset* run and add that info to the output log file. The *submission validator* must flag an error if the `subset` argument is given but the total number of accelerators is not exactly 8, or the model is "8B".

## 4.4. Checkpointing Access Via POSIX API Options

4.4.1. **checkpointPathArgs** --  The arguments to `mlpstorage` that set the directory pathname where the checkpoints are written and read and the directory where the output logfiles are stored must both be set and must be set to different values.

4.4.2. **checkpointFilesystemCheck** --  The `mlpstorage` command should do a "df" command on the directory pathname where the checkpoints are written and read and another one on the directory pathname where the output logfiles are stored and record those values in the logfile.  The *submission validator* should find those entries in the run's logfile and verify that they are different filesystems.  We don't want the submitter to, by acccident, place the logfiles onto the storage system under test since that would skew the results.

## 4.5. Checkpointing Access Via Object API Options

## 4.6.  Checkpointing OPEN versus CLOSED Options

4.6.1. **checkpointClosedMpiProcesses** -- For CLOSED submissions, the number of MPI processes must be set to 8, 64, 512, and 1024 for the respective models.  (see table 2)

4.6.2. **checkpointClosedAcceleratorsPerHost** -- For CLOSED submissions, submitters may adjust the number of simulated accelerators **per host**, as long as each host uses more than 4 simulated accelerators and the total number of simulated accelerators (the total number of processes) matches the requirement.  (see table 2)

4.6.3. **checkpointClosedCheckpointParameters** -- For CLOSED submissions of this benchmark, only a small number of parameters can be modified, and those parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

**Table: Checkpoint Workload Tunable Parameters for CLOSED**

| Parameter                        | Description                                                 | Default               |
|----------------------------------|-------------------------------------------------------------|-----------------------|
| checkpoint.checkpoint_folder     | The storage directory for writing and reading checkpoints   | ./checkpoints/<model> |

4.6.4. **checkpointOpenSubmissionScaling** -- For OPEN submissions of this benchmark, the total number of processes may be increased in multiples of (TP×PP) to showcase the scalability of the storage solution.

**Table 3: Configuration parameters and their mutability in CLOSED and OPEN divisions**

| Parameter                          | Meaning                                      | Default value                                 | Changeable in CLOSED | Changeable in OPEN |
|------------------------------------|----------------------------------------------|-----------------------------------------------|----------------------|--------------------|
| --ppn hostname:slotcount           | Number of processes per node                 | N/A                                           | YES (minimal 4)      | YES (minimal 4)    |
| --num-processes                    | Total number of processes                    | Node local: 8<br>Global: the value in Table 1 | NO                   | YES                |
| --checkpoint-folder                | The folder to save the checkpoint data       | checkpoint/{workload}                         | YES                  | YES                |
| --num-checkpoints-write            | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |
| --num-checkpoints-read             | Number of write checkpoints                  | 10 or 0**                                     | NO                   | NO                 |

**NOTE: In the ``--ppn`` syntax above, the ``slotcount`` value means the number of processes per node to run.**

## 4.7.  Storage System Must Be Simultaneously R/W or _Remappable_

4.7.1. **checkpointCacheFlushValidation** -- If a submitter needs to issue a cache flush operation between the write phase and the read phase of a checkpoint benchmark run, then the validator must check that ``--num-checkpoints-read=0`` was set during the write phase, that there was a short pause of up to 30 seconds maximum, then the write phase was started with ``--num-checkpoints-write=0`` set.

4.7.2. **checkpointTotalTestDuration** -- The validator must verify that the total test duration starts from the timestamp of the first checkpoint written and ends at the ending timestamp of the last checkpoint read, notably including the "remapping" time.

4.7.3. **checkpointRemappingTimeReporting** -- For a _remapping_ solution, the time duration between the checkpoint being completed and the earliest time that that checkpoint could be read by a different host node must be reported in the `SystemDescription.yaml` file.

4.7.4. **checkpointSimultaneousRwSupport** -- The system_configuration.yaml document must list whether the solution support simultaneous reads and/or writes as such:
```
System:
  shared_capabilities:
    multi_host_support: True            # False is used for local storage
    simultaneous_write_support: False   # Are simultaneous writes by multiple hosts supported in the submitted configuration
    simultaneous_read__support: True    # Are simultaneous reads by multiple hosts supported in the submitted configuration
```

# 5.  Validating the VDB Options

## 5.1.  VDB Sizing Options

## 5.2.  VDB Generation Options

## 5.3.  VDB Run Options

## 5.4.  VDB Access Via POSIX API Options

## 5.5.  VDB Access Via Object API Options

## 5.6.  VDB OPEN versus CLOSED Options

# 6.  Validating the KVCache Options

## 6.1.  KVCache Sizing Options

## 6.2.  KVCache Generation Options

## 6.3.  KVCache Run Options

## 6.4.  KVCache Access Via POSIX API Options

## 6.5.  KVCache Access Via Object API Options

## 6.6.  KVCache OPEN versus CLOSED Options
