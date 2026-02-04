# Technology Stack

**Analysis Date:** 2025-01-23

## Languages

**Primary:**
- Python 3.10+ (3.10, 3.11, 3.12 officially supported) - All application code in `mlpstorage/`

**Secondary:**
- YAML - Configuration files in `configs/`, workflow definitions
- Bash/Shell - Wrapper scripts in `kv_cache_benchmark/`, CI workflows

## Runtime

**Environment:**
- CPython 3.10+ (requires `>=3.10.0` per `pyproject.toml`)
- MPI runtime (mpirun/mpiexec) for distributed benchmark execution

**Package Manager:**
- pip with setuptools build backend
- UV (astral.sh) recommended for installation via Ansible
- Lockfile: Not present (uses version ranges)

## Frameworks

**Core:**
- DLIO Benchmark (`dlio-benchmark @ git+https://github.com/argonne-lcf/dlio_benchmark.git@mlperf_storage_v2.0`) - Primary I/O benchmark execution engine
- Hydra (via DLIO) - Configuration management and hierarchical config composition

**Testing:**
- pytest 7.0+ - Test runner
- pytest-cov 4.0+ - Coverage reporting
- pytest-mock 3.0+ - Mocking utilities

**Build/Dev:**
- setuptools 42+ - Build backend
- ruff - Linting and formatting (CI only, not in dev dependencies)

## Key Dependencies

**Critical:**
- `dlio-benchmark` - Core benchmark execution engine, DLIO wraps training/checkpointing workloads
- `mpi4py` (via DLIO) - MPI bindings for distributed execution across cluster nodes
- `pyyaml>=6.0` - Configuration file parsing

**Infrastructure:**
- `psutil>=5.9` - System information collection (CPU, memory, processes)
- `pyarrow` - Data serialization, IPC stream handling
- `yaml` - Configuration loading

**KV Cache Benchmark (separate requirements):**
- `numpy>=1.20.0` - Core numerical operations
- `torch>=2.0.0` - GPU tensor support for KV cache tiers
- `tiktoken>=0.5.0` - OpenAI tokenizer for ShareGPT workload replay
- `pandas>=2.0.0` - Results export
- `openpyxl>=3.1.0` - Excel output support

## Configuration

**Environment Variables:**
- `MLPS_DEBUG` - Enable debug mode globally
- `MPI_RUN_BIN` - Custom mpirun binary path (default: `mpirun`)
- `MPI_EXEC_BIN` - Custom mpiexec binary path (default: `mpiexec`)

**Configuration Files:**
- `pyproject.toml` - Package definition, pytest config, coverage settings
- `configs/dlio/workload/*.yaml` - DLIO workload configurations (unet3d, resnet50, cosmoflow, llama models)
- `configs/vectordbbench/*.yaml` - Vector database benchmark configurations
- `mlpstorage.yaml` - Global options (hosts, data_dir, results_dir)
- `system_configuration.yaml` - Hardware specification template for submissions

**Build Configuration:**
- `pyproject.toml` line 1-3: setuptools build backend

## Platform Requirements

**Development:**
- Python 3.10+ with venv
- MPI development libraries (libopenmpi-dev or mpich-devel)
- Git for DLIO dependency installation

**Production:**
- Linux (Ubuntu 22.04 LTS, RHEL, SLES supported via Ansible)
- MPI runtime (MPICH preferred, OpenMPI supported)
- CUDA toolkit (optional, for KV cache GPU tier)
- Distributed filesystem (NFS, Lustre, etc.) for multi-node benchmarks

**CI/CD:**
- GitHub Actions (`test.yml`)
- Ubuntu latest with Python 3.10, 3.11, 3.12 matrix
- Codecov for coverage reporting

## Installation Methods

**Standard (without DLIO):**
```bash
pip install -e ".[test]"
```

**Full (with DLIO for running benchmarks):**
```bash
pip install -e ".[full]"
```

**Ansible Deployment:**
- Uses UV package manager
- Installs from GitHub main branch
- Configures MPI environment (MPICH)

---

*Stack analysis: 2025-01-23*
