
# Capturing TODO Items:
 [ ] Configure datasize to collect the memory information from the hosts instead of getting a number of hosts for the calculation

 [ ] Determine method to use cgroups for memory limitation in the benchmark script.

 [ ] Add a log block at the start of datagen & run that output all the parms being used to be clear on what a run is.

 [ ] Remove accelerator type from datagen
 [ ] datasize should output the datagen command to copy and paste

 [ ] Add autosize parameter for run_benchmark and datasize
 [ ] for run it's just size of dataset based on memory capacity
 [ ] For datasize it needs an input of GB/s for the cluster and list of hosts

 [ ] Keep a log of mlperfstorage commands executed in a mlperf.history file in results_dir

 [ ] Add support for datagen to use subdirectories
 [ ] Capture cluster information and write to a json document in outputdir. Figure out how to get all clients for milvus