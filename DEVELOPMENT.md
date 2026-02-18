# Rules Updates

- [ ] Define filesystem caching rules in detail
- [ ] Define system json schema and creation process
- [ ] Define allowed time between runs
- [ ] Define rules that use local SSD for caching data
- [ ] Define rules for hyperconverged and local cache

# Code Updates
- [ ] Configure datasize to collect the memory information from the hosts instead of getting a number of hosts for the calculation

- [ ] Determine method to use cgroups for memory limitation in the benchmark script.

- [x] Add a log block at the start of datagen & run that output all the parms being used to be clear on what a run is.

- [x] Remove accelerator type from datagen
- [x] datasize should output the datagen command to copy and paste

- [ ] Add autosize parameter for run_benchmark and datasize
- [ ] for run it's just size of dataset based on memory capacity
- [ ] For datasize it needs an input of GB/s for the cluster and list of hosts
-
- [x] Keep a log of mlperfstorage commands executed in a mlperf.history file in results_dir

- [ ] Add support for datagen to use subdirectories
- [x] Capture cluster information and write to a json document in outputdir. 
- [ ] Figure out how to get all clients for milvus

## benchmark[.py | .sh] script
- [x] Unique names for files and directories with structure for benchmark, accelerator, count, run-sequence, run-number
- [x] Better installer that manages dependencies
- [ ] Containerization
- - [ ] Ease of Deployment of Benchmark (just get it working)
- - [ ] Cgroups and resource limits (better cache management)
- [ ] Flush Cache before a run
- [ ] Validate inputs for –closed runs (eg: don’t allow runs against datasets that are too small)
- [ ] Reportgen should run validation against outputs
- [ ] Add better system.json creation to automate the system description for consistency
- - [ ] Add json schema checker for system documents that submitters create
- [ ] Automate execution of multiple runs
- [ ] ~~Add support for code changes in closed to supported categories [ data loader, s3 connector, etc]~~
- - [ ] ~~Add patches directory that gets applied before execution~~
- [ ] Add runtime estimation 
- [x] and --what-if or --dry-run flag
- [ ] Automate selection of minimum required dataset
- [ ] ~~Determine if batch sizes in MLPerf Training are representative of batch sizes for realistically sized datasets~~
- [ ] Split system.json into automatically capturable (clients) and manual (storage)
- [ ] Define system.json schema and add schema checker to the tool for reportgen
- [ ] Add report-dir csv of results from tests as they are run
- [ ] Collect versions of all prerequisite packages for storage and dlio

## DLIO Improvements
- [ ] Reduce verbosity of logging
- [ ] Add callback handler for custom monitoring
- - [ ] SPECStorage uses a “PRIME_MON_SCRIPT” environment variable that will execute at different times
- - [ ] Checkpoint_bench uses RPC to call execution which can be wrapped externally
- [ ] Add support for DIRECTIO
- [ ] Add seed for dataset creation so that distribution of sizes is the same for all submitters (file 1 = mean + x bytes, file 2 = mean + y bytes, etc)
- [ ] Determine if global barrier for each batch matches industry behavior

## Results Presentation
- [ ] Better linking and presentation of system diagrams (add working links to system diagrams to supplementals)
- [ ] Define presentation and rules for hyperconverged or systems with local cache