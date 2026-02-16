# Deep Learning I/O (DLIO) Benchmark
![test status](https://github.com/argonne-lcf/dlio_benchmark/actions/workflows/ci.yml/badge.svg)

This README provides an abbreviated documentation of the DLIO code. Please refer to https://dlio-benchmark.readthedocs.io for full user documentation. 

## Overview

DLIO is an I/O benchmark for Deep Learning. DLIO is aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. 

## Installation and running DLIO
### Bare metal installation 

```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
pip install .
dlio_benchmark ++workload.workflow.generate_data=True
```

### Bare metal installation with profiler

```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
pip install .[pydftracer]
```

## Container
```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
docker build -t dlio .
docker run -t dlio dlio_benchmark ++workload.workflow.generate_data=True
``` 

You can also pull rebuilt container from docker hub (might not reflect the most recent change of the code): 
```bash
docker pull docker.io/zhenghh04/dlio:latest
docker run -t docker.io/zhenghh04/dlio:latest dlio_benchmark ++workload.workflow.generate_data=True
```
If your running on a different architecture, refer to the Dockerfile to build the dlio_benchmark container from scratch.

One can also run interactively inside the container
```bash
docker run -t docker.io/zhenghh04/dlio:latest /bin/bash
root@30358dd47935:/workspace/dlio$ dlio_benchmark ++workload.workflow.generate_data=True
```

## PowerPC
PowerPC requires installation through anaconda.
```bash
# Setup required channels
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

# create and activate environment
conda env create --prefix ./dlio_env_ppc --file environment-ppc.yaml --force
conda activate ./dlio_env_ppc
# install other dependencies
python -m pip install .
```

## Lassen, LLNL
For specific instructions on how to install and run the benchmark on Lassen please refer to: [Install Lassen](https://dlio-benchmark.readthedocs.io/en/latest/instruction_lassen.html)

## Running the benchmark

A DLIO run is split in 3 phases: 
- Generate synthetic data that DLIO will use
- Run the benchmark using the previously generated data
- Post-process the results to generate a report

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be found in [dlio_benchmark/configs/workload/](./dlio_benchmark/configs/workload). 

One can specify the workload through the ```workload=``` option on the command line. Specific configuration fields can then be overridden following the ```hydra``` framework convention (e.g. ```++workload.framework=tensorflow```). 

First, generate the data
  ```bash
  mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False
  ```
If possible, one can flush the filesystem caches in order to properly capture device I/O
  ```bash
  sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
  ```
Finally, run the benchmark
  ```bash
  mpirun -np 8 dlio_benchmark workload=unet3d
  ```
Finally, run the benchmark with Tracer
  ```bash
  export DFTRACER_ENABLE=1
  export DFTRACER_INC_METADATA=1
  mpirun -np 8 dlio_benchmark workload=unet3d
  ```

All the outputs will be stored in ```hydra_log/unet3d/$DATE-$TIME``` folder. To post process the data, one can do
```bash 
dlio_postprocessor --output-folder hydra_log/unet3d/$DATE-$TIME
```
This will generate ```DLIO_$model_report.txt``` in the output folder. 

## Workload YAML configuration file 
Workload characteristics are specified by a YAML configuration file. Below is an example of a YAML file for the UNet3D workload which is used for 3D image segmentation. 

```
# contents of unet3d.yaml
model: 
  name: unet3d
  model_size: 499153191

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: True

dataset: 
  data_folder: data/unet3d/
  format: npz
  num_files_train: 168
  num_samples_per_file: 1
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
  record_length_bytes_resize: 2097152
  
reader: 
  data_loader: pytorch
  batch_size: 4
  read_threads: 4
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 5
  computation_time: 1.3604

checkpoint:
  checkpoint_folder: checkpoints/unet3d
  checkpoint_after_epoch: 5
  epochs_between_checkpoints: 2
```

The full list of configurations can be found in: https://argonne-lcf.github.io/dlio_benchmark/config.html

The YAML file is loaded through hydra (https://hydra.cc/). The default setting are overridden by the configurations loaded from the YAML file. One can override the configuration through command line (https://hydra.cc/docs/advanced/override_grammar/basic/). 

## Current Limitations and Future Work

* DLIO currently assumes the samples to always be 2D images, even though one can set the size of each sample through ```--record_length```. We expect the shape of the sample to have minimal impact to the I/O itself. This yet to be validated for case by case perspective. We plan to add option to allow specifying the shape of the sample. 

* We assume the data/label pairs are stored in the same file. Storing data and labels in separate files will be supported in future.

* File format support: we only support tfrecord, hdf5, npz, csv, jpg, jpeg formats. Other data formats can be extended. 

* Data Loader support: we support reading datasets using TensorFlow tf.data data loader, PyTorch DataLoader, and a set of custom data readers implemented in ./reader. For TensorFlow tf.data data loader, PyTorch DataLoader  
  - We have complete support for tfrecord format in TensorFlow data loader. 
  - For npz, jpg, jpeg, hdf5, we currently only support one sample per file case. In other words, each sample is stored in an independent file. Multiple samples per file case will be supported in future. 

## How to contribute 
We welcome contributions from the community to the benchmark code. Specifically, we welcome contribution in the following aspects:
General new features needed including: 

* support for new workloads: if you think that your workload(s) would be interested to the public, and would like to provide the yaml file to be included in the repo, please submit an issue.  
* support for new data loaders, such as DALI loader, MxNet loader, etc
* support for new frameworks, such as MxNet
* support for noval file systems or storage, such as AWS S3. 
* support for loading new data formats. 

If you would like to contribute, please submit an issue to https://github.com/argonne-lcf/dlio_benchmark/issues, and contact ALCF DLIO team, Huihuo Zheng at huihuo.zheng@anl.gov

## Citation and Reference
The original CCGrid'21 paper describes the design and implementation of DLIO code. Please cite this paper if you use DLIO for your research. 

```
@article{devarajan2021dlio,
  title={DLIO: A Data-Centric Benchmark for Scientific Deep Learning Applications},
  author={H. Devarajan and H. Zheng and A. Kougkas and X.-H. Sun and V. Vishwanath},
  booktitle={IEEE/ACM International Symposium in Cluster, Cloud, and Internet Computing (CCGrid'21)},
  year={2021},
  volume={},
  number={81--91},
  pages={},
  publisher={IEEE/ACM}
}
```

We also encourage people to take a look at a relevant work from MLPerf Storage working group. 
```
@article{balmau2022mlperfstorage,
  title={Characterizing I/O in Machine Learning with MLPerf Storage},
  author={O. Balmau},
  booktitle={SIGMOD Record DBrainstorming},
  year={2022},
  volume={51},
  number={3},
  publisher={ACM}
}
```

## Acknowledgments

This work used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility under Contract DE-AC02-06CH11357 and is supported in part by National Science Foundation under NSF, OCI-1835764 and NSF, CSR-1814872.

## License

Apache 2.0 [LICENSE](./LICENSE)

---------------------------------------
Copyright (c) 2025, UChicago Argonne, LLC
All Rights Reserved

If you have questions about your rights to use or distribute this software, please contact Argonne Intellectual Property Office at partners@anl.gov

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
