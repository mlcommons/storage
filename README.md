# MLPerf Storage Benchmark Suite
MLPerf® Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Configuration](#configuration)
- [Workload Categories](#workload-categories)
- [Submission Rules](#submission-rules)

---

## Documentation

Two README files cover the full project in detail — read both before diving into the
code or running benchmarks:

| Document | What it covers |
|----------|----------------|
| **[docs/README.md](docs/README.md)** | Complete project overview: all four benchmark workloads, document reference, object storage library guides, and quick-link index to every test script |
| **[tests/README.md](tests/README.md)** | Everything needed to run tests: environment setup, unit tests, integration tests, object-store performance scripts, and how pytest is configured |

The top-level sections below give the official MLCommons parameter reference and
are retained for submission compliance.

---

## Overview
For an overview of how this benchmark suite is used by submitters to compare the performance of storage systems supporting an AI cluster, see the MLPerf® Storage Benchmark submission rules here: [doc](https://github.com/mlcommons/storage/blob/main/Submission_guidelines.md). 

## Prerequisite

The installation and the configuration steps described in this README are validated against clients running Ubuntu 24.04 server with python 3.12.3. The benchmark script has to be run only in one participating client host(any) which internally calls `mpirun` to launch the distributed workloads across multiple client hosts. The launcher client host also participates in the distributed training process.

Following prerequisites must be satisfied

1. Pick one host to act as the launcher client host. Passwordless ssh must be setup from the launcher client host to all other participating client hosts.  `ssh-copy-id` is a useful tool.
2. The code and data location(discussed in further sections) must be exactly same in every client host including the launcher host. This is because, the same benchmark command is automatically triggered in every participating client host during the distributed training process.

## Installation 
**The following installation steps must be run on every client host that will participate in running the benchmarks.**

### uv (Required)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager that handles virtual environment creation, dependency resolution, and Python version management automatically — no manual `venv` or `pip` steps required. It will install into your virutal environment exactly the versions of supporting libraries and tools that the benchmark has been tested with.

**Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the MPI runtime (still required for distributed execution):

```bash
sudo apt install libopenmpi-dev openmpi-common
```

Clone the repo:

```bash
git clone https://github.com/mlcommons/storage.git
cd storage
```

Verify the installation:

```bash
mlpstorage --help
```

The `mlpstorage` script executes `uv run` every time you invoke the benchmark, keeping your virtual environment up to date.
`uv` creates a `.venv` virtual environment and installs all dependencies — including DLIO benchmark — automatically based upon the contents of the the `uv.lock` file.

> **Note:** `uv` installs the CPU-only version of PyTorch.
> GPU-accelerated training or checkpointing workloads are not supported, there is no need to have GPUs in your benchmark test gear, they will not be used.

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/mlcommons/DLIO_local_changes) code, a benchmark suite for emulating I/O patterns for deep learning workloads. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `configs` folder. 

### Testing the Installation

See **[tests/README.md](tests/README.md)** for the complete test guide — environment
setup, unit tests (no infrastructure required), integration tests, and object-store
performance scripts for all three supported object storage libraries.

- **StreamingCheckpointing Demo**: Run `./tests/checkpointing/demo_checkpoint_methods.sh` to see:
  - dgen-py integration (155× faster data generation)
  - StreamingCheckpointing (192× memory reduction)
  - Comparison of old vs new checkpoint methods

- **Backend Validation**: Test multi-library support:
  ```bash
  python tests/checkpointing/test_streaming_backends.py --backends s3dlio minio
  ```

- **Unit tests** (no infrastructure required):
  ```bash
  pytest tests/unit/
  ```

### Workload Categories
The benchmark uses nested commands to select the workload category, workload, and workload parameters.
The first argument is the workload category:
 - training
 - checkpointing
 - vectordb
 - kvcache

```bash
[root@localhost ]#  mlpstorage -h
usage: mlpstorage [-h] [--version] {training,checkpointing,vectordb,kvcache} ...

Script to launch the MLPerf Storage benchmark

positional arguments:
  {training,checkpointing,vectordb,kvcache}
    training            Training benchmark options
    checkpointing       Checkpointing benchmark options
    vectordb            VectorDB benchmark options
    kvcache             KVCcache benchmark options

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### Training Category
The training category supports emulation of the training of 3 models (FLUX.1, RetinaNet, and DLRMv2).

See [training/README.md](training/README.md) for more details.

### Checkpointing Category
The checkpointing category supports emulation of taking a checkpoint of an LLM foundation training task,
specifically the Llama3 LLM at four different scales: 8B, 70B, 405B, and 1250B parameters.

See [checkpointing/README.md](checkpointing/README.md) for more details.

### VectorDB Category
The vectordb category supports emulation of a vector database as used in an LLM RAG pipeline,
specifically the Milvus VDB using one of three different algorithms: DiskANN, HNSW, and AiSAQ.

See [vdb_benchmark/README.md](vdb_benchmark/README.md) for more details.

### KVCache Category
The kvcache category supports emulation of a context cache as used by an LLM.

See [kv_cache_benchmark/README.md](kv_cache_benchmark/README.md) for more details.

## Submission Rules

MLPerf™ Storage Benchmark submission rules are described in the
[Rules.md](https://github.com/mlcommons/storage/blob/main/Rules.md) file.
If you have questions, please contact the [Storage WG chairs](https://mlcommons.org/en/groups/research-storage/).
