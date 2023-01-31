# MLPerf™ Storage Benchmark Suite
MLPerf Storage is a benchmark suite to characterize performance of storage systems that support machine learning workloads.

- [Overview](#Overview) 
- [Installation](#Installation)
- [Configuration](#Configuration)
- [Workloads](#Workloads)
	- [U-Net3D](#U-Net3D)
	- [BERT](#BERT) 
	- [DLRM](#DLRM)
- [Parameters](#Parameters)      
- [Releases](#Releases)
## Overview

To be added

## Installation 

Install dependencies using your system package manager.
- `mpich` for MPI package
- `sysstat` for iostat package

For eg: when running on Ubuntu OS,

```
sudo apt-get install mpich sysstat
```

Clone the [MLCommons Storage](https://github.com/mlcommons/storage) repository and install Python dependencies.

```bash
git clone --recurse-submodules https://github.com/mlcommons/storage.git 
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
               |---unet3d.yaml
               |---bert.yaml
```

The configuration of each workload is specified through a yaml file. You can see the configs of all workloads in `storage-conf` folder.

```bash
./benchmark.sh -h

Usage: ./benchmark.sh [datagen/run/configview/reportgen] [options]
Script to launch the MLPerf Storage benchmark.
```
## Configuration

The benchmark suite consists of 3 distinct phases

1. Synthetic data is generated based on the workload requested by the user.

```bash
./benchmark.sh datagen -h

Usage: ./benchmark.sh datagen [options]
Generate benchmark dataset based on the specified options.


Options:
  -h, --help			Print this message
  -c, --category		Benchmark category to be submitted. Possible options are 'closed'(default)
  -w, --workload		Workload dataset to be generated. Possible options are 'unet3d', 'bert'
  -n, --num-parallel		Number of parallel jobs used to generate the dataset
  -r, --results-dir		Location to the results directory
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For generating training data for `unet3d` workload into `unet3d_data` directory with 10 subfolders using 8 parallel jobs, 

```bash
./benchmark.sh datagen --workload unet3d --num-parallel 8 --param dataset.num_subfolders_train=10 --param dataset.data_folder=unet3d_data
```

2. Benchmark is run on the generated data. Device stats are collected continuously using iostat profiler during the benchmark run.

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
  -r, --results-dir		Location to the results directory
  -p, --param			DLIO param when set, will override the config file value
```

Example:

For running benchmark on `unet3d` workload with data located in `unet3d_data` directory using 4 accelerators and results on `unet3d_results` directory , 

```bash
./benchmark.sh run --workload unet3d --num-accelerators 4 --results-dir unet3d_results --param dataset.data_folder=unet3d_data
```

3. Reports are generated from the benchmark results

```bash
./benchmark.sh reportgen -h

Usage: ./benchmark.sh reportgen [options]
Generate a report from the benchmark results.


Options:
  -h, --help			Print this message
  -r, --results-dir		Location to the results directory
```

## Workloads
Currently, the storage benchmark suite supports benchmarking of 3 deep learning workloads
- Image segmentation using U-Net3D model ([unet3d](./storage-conf/workloads/unet3d.yaml))
- Natural language processing using BERT model ([bert](./storage-conf/workloads/bert.yaml))
- Recommendation using DLRM model (TODO)

### U-Net3D Workload

Generate data for the benchmark run

```bash
./benchmark.sh datagen --workload unet3d --num-parallel 8
```
  
Flush the filesystem caches before benchmark run in order to properly capture device I/O

```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```
  
Run the benchmark.

```bash
./benchmark.sh run --workload unet3d --num-accelerators 8
```

All results will be stored in ```results/unet3d/$DATE-$TIME``` folder or in the directory when overriden using `--results-dir`(or `-r`) argument. To generate the final report, one can do

```bash 
./benchmark.sh reportgen --results-dir results/unet3d/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` in the output folder. 

### BERT Workload

Generate data for the benchmark run

```bash
./benchmark.sh datagen --workload bert --num-parallel 8
```
  
Flush the filesystem caches before benchmark run in order to properly capture device I/O
```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```
  
Run the benchmark
```bash
./benchmark.sh run --workload bert --num-accelerators 8
```

All results will be stored in ```results/bert/$DATE-$TIME``` folder or in the directory when overriden using `--results-dir`(or `-r`) argument. To generate the final report, one can do

```bash 
./benchmark.sh reportgen -r results/bert/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` in the output folder. 


### DLRM Workload

To be added

## Parameters 

Below table displays the list of configurable paramters for the benchmark. 

| Parameter                      | Description                                                 |Default|
| ------------------------------ | ------------------------------------------------------------ |-------|
| **Dataset params**		|								|   |
| dataset.num_files_train       | Number of files for the training set  		        | --|
| dataset.num_subfolders_train  | Number of subfolders that the training set is stored	        |0|
| dataset.data_folder           | The path where dataset is stored				| --|
| dataset.keep_files  		| Flag whether to keep the dataset files afer the run	        |True|
| **Reader params**				|						|   |
| reader.read_threads		| Number of threads to load the data                            | --|
| reader.computation_threads    | Number of threads to preprocess the data(only for bert)       | --|
| reader.prefetch_size		| Number of batch to prefetch 			                |0|
| **Checkpoint params**		|								|   |
| checkpoint.checkpoint_folder	| The folder to save the checkpoints  				| --|
| **Storage params**		|								|   |
| storage.storage_root		| The storage root directory  					| ./|
| storage.storage_type		| The storage type  						|local_fs|
=======
## Overview

Storage benchmark suite uses DLIO as a submodule inorder to 

1. Generate synthetic data
2. Run the benchmark on the generated data
3. Create reports from the benchmark results

## Releases
The benchmark preview package will be released soon. Stay tuned!

